# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This software is dual licensed under the GNU General Public License and also
# under a 3-clause BSD license. Recipients may choose which of these licenses
# to use; please see the files gpl-3.0.txt and/or bsd_license.txt,
# respectively.  If you choose the GPL option then the following text applies
# (but note that there is still no warranty even if you opt for BSD instead):
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid

__all__ = ("DirectQueryDriver",)

import dataclasses
import logging
import sys
from collections.abc import Iterable, Mapping, Set
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, cast, overload

import sqlalchemy

from .. import ddl
from .._dataset_type import DatasetType
from .._exceptions import InvalidQueryError
from ..dimensions import DataIdValue, DimensionGroup, DimensionRecordSet, DimensionUniverse, SkyPixDimension
from ..name_shrinker import NameShrinker
from ..queries import tree as qt
from ..queries.driver import (
    DataCoordinateResultPage,
    DatasetRefResultPage,
    DimensionRecordResultPage,
    GeneralResultPage,
    PageKey,
    QueryDriver,
    ResultPage,
)
from ..queries.result_specs import (
    DataCoordinateResultSpec,
    DatasetRefResultSpec,
    DimensionRecordResultSpec,
    GeneralResultSpec,
    ResultSpec,
)
from ..registry import CollectionSummary, CollectionType, NoDefaultCollectionError, RegistryDefaults
from ..registry.interfaces import ChainedCollectionRecord, CollectionRecord
from ..registry.managers import RegistryManagerInstances
from ._postprocessing import Postprocessing
from ._query_builder import QueryBuilder, QueryJoiner
from ._query_plan import (
    QueryFindFirstPlan,
    QueryJoinsPlan,
    QueryPlan,
    QueryProjectionPlan,
    ResolvedDatasetSearch,
)
from ._sql_column_visitor import SqlColumnVisitor

if TYPE_CHECKING:
    from ..registry.interfaces import Database


_LOG = logging.getLogger(__name__)


class DirectQueryDriver(QueryDriver):
    """The `QueryDriver` implementation for `DirectButler`.

    Parameters
    ----------
    db : `Database`
        Abstraction for the SQL database.
    universe : `DimensionUniverse`
        Definitions of all dimensions.
    managers : `RegistryManagerInstances`
        Struct of registry manager objects.
    defaults : `RegistryDefaults`
        Struct holding the default collection search path and governor
        dimensions.
    raw_page_size : `int`, optional
        Number of database rows to fetch for each result page.  The actual
        number of rows in a page may be smaller due to postprocessing.
    constant_rows_limit : `int`, optional
        Maximum number of uploaded rows to include in queries via
        `Database.constant_rows`; above this limit a temporary table is used
        instead.
    postprocessing_filter_factor : `int`, optional
        The number of database rows we expect to have to fetch to yield a
        single output row for queries that involve postprocessing.  This is
        purely a performance tuning parameter that attempts to balance between
        fetching too much and requiring multiple fetches; the true value is
        highly dependent on the actual query.
    """

    def __init__(
        self,
        db: Database,
        universe: DimensionUniverse,
        managers: RegistryManagerInstances,
        defaults: RegistryDefaults,
        raw_page_size: int = 10000,
        constant_rows_limit: int = 1000,
        postprocessing_filter_factor: int = 10,
    ):
        self.db = db
        self.managers = managers
        self._universe = universe
        self._defaults = defaults
        self._materializations: dict[qt.MaterializationKey, _MaterializationState] = {}
        self._upload_tables: dict[qt.DataCoordinateUploadKey, sqlalchemy.FromClause] = {}
        self._exit_stack: ExitStack | None = None
        self._raw_page_size = raw_page_size
        self._postprocessing_filter_factor = postprocessing_filter_factor
        self._constant_rows_limit = constant_rows_limit
        self._cursors: dict[PageKey, _Cursor] = {}

    def __enter__(self) -> None:
        self._exit_stack = ExitStack()
        # It might be nice to defer opening a transaction here until first use
        # to reduce the time spent in transactions.  But it's worth noting that
        # this is the default low-level behavior of the Python SQLite driver,
        # and it makes it incredibly prone to deadlocks.  We might be okay
        # here, because Query doesn't do true write operations - just temp
        # table writes - but I'm not confident that's enough to make delayed
        # transaction starts safe against deadlocks, and it'd be more
        # complicated to implement anyway.
        #
        # We start a transaction rather than just opening a connection to make
        # temp table and cursors work with pg_bouncer transaction affinity.
        self._exit_stack.enter_context(self.db.transaction(for_temp_tables=True))

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        assert self._exit_stack is not None
        self._materializations.clear()
        self._upload_tables.clear()
        while self._cursors:
            _, cursor = self._cursors.popitem()
            cursor.close(exc_type, exc_value, traceback)
        self._exit_stack.__exit__(exc_type, exc_value, traceback)
        self._exit_stack = None

    @property
    def universe(self) -> DimensionUniverse:
        return self._universe

    @overload
    def execute(
        self, result_spec: DataCoordinateResultSpec, tree: qt.QueryTree
    ) -> DataCoordinateResultPage: ...

    @overload
    def execute(
        self, result_spec: DimensionRecordResultSpec, tree: qt.QueryTree
    ) -> DimensionRecordResultPage: ...

    @overload
    def execute(self, result_spec: DatasetRefResultSpec, tree: qt.QueryTree) -> DatasetRefResultPage: ...

    @overload
    def execute(self, result_spec: GeneralResultSpec, tree: qt.QueryTree) -> GeneralResultPage: ...

    def execute(self, result_spec: ResultSpec, tree: qt.QueryTree) -> ResultPage:
        # Docstring inherited.
        if self._exit_stack is None:
            raise RuntimeError("QueryDriver context must be entered before queries can be executed.")
        _, builder = self.build_query(
            tree,
            final_columns=result_spec.get_result_columns(),
            order_by=result_spec.order_by,
            find_first_dataset=result_spec.find_first_dataset,
        )
        sql_select = builder.select()
        if result_spec.order_by:
            visitor = SqlColumnVisitor(builder.joiner, self)
            sql_select = sql_select.order_by(*[visitor.expect_scalar(term) for term in result_spec.order_by])
        if result_spec.limit is not None:
            if builder.postprocessing:
                builder.postprocessing.limit = result_spec.limit
            else:
                sql_select = sql_select.limit(result_spec.limit)
        if builder.postprocessing.limit is not None:
            # We might want to fetch many fewer rows than the default page
            # size if we have to implement limit in postprocessing.
            raw_page_size = min(
                self._postprocessing_filter_factor * builder.postprocessing.limit,
                self._raw_page_size,
            )
        else:
            raw_page_size = self._raw_page_size
        # Execute the query by initializing a _Cursor object that manages the
        # lifetime of the result.
        cursor = _Cursor(
            self.db,
            sql_select,
            result_spec=result_spec,
            name_shrinker=builder.joiner.name_shrinker,
            postprocessing=builder.postprocessing,
            raw_page_size=raw_page_size,
        )
        result_page = cursor.next()
        if result_page.next_key is not None:
            # Cursor has not been exhausted; add it to the driver for use by
            # fetch_next_page.
            self._cursors[result_page.next_key] = cursor
        return result_page

    @overload
    def fetch_next_page(
        self, result_spec: DataCoordinateResultSpec, key: PageKey
    ) -> DataCoordinateResultPage: ...

    @overload
    def fetch_next_page(
        self, result_spec: DimensionRecordResultSpec, key: PageKey
    ) -> DimensionRecordResultPage: ...

    @overload
    def fetch_next_page(self, result_spec: DatasetRefResultSpec, key: PageKey) -> DatasetRefResultPage: ...

    @overload
    def fetch_next_page(self, result_spec: GeneralResultSpec, key: PageKey) -> GeneralResultPage: ...

    def fetch_next_page(self, result_spec: ResultSpec, key: PageKey) -> ResultPage:
        # Docstring inherited.
        try:
            cursor = self._cursors.pop(key)
        except KeyError:
            raise RuntimeError("Cannot continue query result iteration after the query context has closed.")
        result_page = cursor.next()
        if result_page.next_key is not None:
            self._cursors[result_page.next_key] = cursor
        return result_page

    def materialize(
        self,
        tree: qt.QueryTree,
        dimensions: DimensionGroup,
        datasets: frozenset[str],
    ) -> qt.MaterializationKey:
        # Docstring inherited.
        if self._exit_stack is None:
            raise RuntimeError("QueryDriver context must be entered before 'materialize' is called.")
        _, builder = self.build_query(tree, qt.ColumnSet(dimensions))
        # Current implementation ignores 'datasets' aside from remembering
        # them, because figuring out what to put in the temporary table for
        # them is tricky, especially if calibration collections are involved.
        # That's okay because:
        #
        # - the query whose results we materialize includes the dataset
        #   searches as constraints;
        #
        # - we still (in Query.materialize) join the dataset searches back in
        #   anyway, and given materialized data IDs the join to the dataset
        #   search is straightforward and definitely well-indexed, and not much
        #   (if at all) worse than joining back in on a materialized UUID.
        #
        sql_select = builder.select()
        table = self._exit_stack.enter_context(self.db.temporary_table(builder.make_table_spec()))
        self.db.insert(table, select=sql_select)
        key = uuid.uuid4()
        self._materializations[key] = _MaterializationState(table, datasets, builder.postprocessing)
        return key

    def upload_data_coordinates(
        self, dimensions: DimensionGroup, rows: Iterable[tuple[DataIdValue, ...]]
    ) -> qt.DataCoordinateUploadKey:
        # Docstring inherited.
        if self._exit_stack is None:
            raise RuntimeError(
                "QueryDriver context must be entered before 'upload_data_coordinates' is called."
            )
        columns = qt.ColumnSet(dimensions).drop_implied_dimension_keys()
        table_spec = ddl.TableSpec(
            [columns.get_column_spec(logical_table, field).to_sql_spec() for logical_table, field in columns]
        )
        dict_rows: list[dict[str, Any]]
        if not columns:
            table_spec.fields.add(
                ddl.FieldSpec(
                    QueryBuilder.EMPTY_COLUMNS_NAME, dtype=QueryBuilder.EMPTY_COLUMNS_TYPE, nullable=True
                )
            )
            dict_rows = [{QueryBuilder.EMPTY_COLUMNS_NAME: None}]
        else:
            dict_rows = [dict(zip(dimensions.required, values)) for values in rows]
        from_clause: sqlalchemy.FromClause
        if len(dict_rows) > self._constant_rows_limit:
            from_clause = self._exit_stack.enter_context(self.db.temporary_table(table_spec))
            self.db.insert(from_clause, *dict_rows)
        else:
            from_clause = self.db.constant_rows(table_spec.fields, *dict_rows)
        key = uuid.uuid4()
        self._upload_tables[key] = from_clause
        return key

    def count(
        self,
        tree: qt.QueryTree,
        result_spec: ResultSpec,
        *,
        exact: bool,
        discard: bool,
    ) -> int:
        # Docstring inherited.
        columns = result_spec.get_result_columns()
        plan, builder = self.build_query(tree, columns, find_first_dataset=result_spec.find_first_dataset)
        if not all(d.collection_records for d in plan.joins.datasets.values()):
            return 0
        if not exact:
            builder.postprocessing = Postprocessing()
        if builder.postprocessing:
            if not discard:
                raise RuntimeError("Cannot count query rows exactly without discarding them.")
            sql_select = builder.select()
            builder.postprocessing.limit = result_spec.limit
            n = 0
            with self.db.query(sql_select.execution_options(yield_per=self._raw_page_size)) as results:
                for _ in builder.postprocessing.apply(results):
                    n += 1
            return n
        # If the query has DISTINCT or GROUP BY, nest it in a subquery so we
        # count deduplicated rows.
        builder = builder.nested()
        # Replace the columns of the query with just COUNT(*).
        builder.columns = qt.ColumnSet(self._universe.empty.as_group())
        count_func: sqlalchemy.ColumnElement[int] = sqlalchemy.func.count()
        builder.joiner.special["_ROWCOUNT"] = count_func
        # Render and run the query.
        sql_select = builder.select()
        with self.db.query(sql_select) as result:
            count = cast(int, result.scalar())
        if result_spec.limit is not None:
            count = min(count, result_spec.limit)
        return count

    def any(self, tree: qt.QueryTree, *, execute: bool, exact: bool) -> bool:
        # Docstring inherited.
        plan, builder = self.build_query(tree, qt.ColumnSet(tree.dimensions))
        if not all(d.collection_records for d in plan.joins.datasets.values()):
            return False
        if not execute:
            if exact:
                raise RuntimeError("Cannot obtain exact result for 'any' without executing.")
            return True
        if builder.postprocessing and exact:
            sql_select = builder.select()
            with self.db.query(
                sql_select.execution_options(yield_per=self._postprocessing_filter_factor)
            ) as result:
                for _ in builder.postprocessing.apply(result):
                    return True
                return False
        sql_select = builder.select().limit(1)
        with self.db.query(sql_select) as result:
            return result.first() is not None

    def explain_no_results(self, tree: qt.QueryTree, execute: bool) -> Iterable[str]:
        # Docstring inherited.
        plan, _ = self.analyze_query(tree, qt.ColumnSet(tree.dimensions))
        if plan.joins.messages or not execute:
            return plan.joins.messages
        # TODO: guess at ways to split up query that might fail or succeed if
        # run separately, execute them with LIMIT 1 and report the results.
        return []

    def get_dataset_type(self, name: str) -> DatasetType:
        # Docstring inherited
        return self.managers.datasets[name].datasetType

    def get_default_collections(self) -> tuple[str, ...]:
        # Docstring inherited.
        if not self._defaults.collections:
            raise NoDefaultCollectionError("No collections provided and no default collections.")
        return tuple(self._defaults.collections)

    def build_query(
        self,
        tree: qt.QueryTree,
        final_columns: qt.ColumnSet,
        order_by: Iterable[qt.OrderExpression] = (),
        find_first_dataset: str | None = None,
    ) -> tuple[QueryPlan, QueryBuilder]:
        """Convert a query description into a mostly-completed `QueryBuilder`.

        Parameters
        ----------
        tree : `.queries.tree.QueryTree`
            Description of the joins and row filters in the query.
        final_columns : `.queries.tree.ColumnSet`
            Final output columns that should be emitted by the SQL query.
        order_by : `~collections.abc.Iterable` [ \
                `.queries.tree.OrderExpression` ], optional
            Column expressions to sort by.
        find_first_dataset : `str` or `None`, optional
            Name of a dataset type for which only one result row for each data
            ID should be returned, with the colletions searched in order.

        Returns
        -------
        plan : `QueryPlan`
            Plan used to transform the query into SQL, including some
            information (e.g. diagnostics about doomed-to-fail dataset
            searches) that isn't transferred into the builder itself.
        builder : `QueryBuilder`
            Builder object that can be used to create a SQL SELECT via its
            `~QueryBuilder.select` method.  We return this instead of a
            `sqlalchemy.Select` object itself to allow different methods to
            customize the SELECT clause itself (e.g. `count` can replace the
            columns selected with ``COUNT(*)``).
        """
        # See the QueryPlan docs for an overview of what these stages of query
        # construction do.
        plan, builder = self.analyze_query(tree, final_columns, order_by, find_first_dataset)
        self.apply_query_joins(plan.joins, builder.joiner)
        self.apply_query_projection(plan.projection, builder)
        builder = self.apply_query_find_first(plan.find_first, builder)
        builder.columns = plan.final_columns
        return plan, builder

    def analyze_query(
        self,
        tree: qt.QueryTree,
        final_columns: qt.ColumnSet,
        order_by: Iterable[qt.OrderExpression] = (),
        find_first_dataset: str | None = None,
    ) -> tuple[QueryPlan, QueryBuilder]:
        """Construct a plan for building a query and initialize a builder.

        Parameters
        ----------
        tree : `.queries.tree.QueryTree`
            Description of the joins and row filters in the query.
        final_columns : `.queries.tree.ColumnSet`
            Final output columns that should be emitted by the SQL query.
        order_by : `~collections.abc.Iterable` [ \
                `.queries.tree.OrderExpression` ], optional
            Column expressions to sort by.
        find_first_dataset : `str` or `None`, optional
            Name of a dataset type for which only one result row for each data
            ID should be returned, with the colletions searched in order.

        Returns
        -------
        plan : `QueryPlan`
            Plan used to transform the query into SQL, including some
            information (e.g. diagnostics about doomed-to-fail dataset
            searches) that isn't transferred into the builder itself.
        builder : `QueryBuilder`
            Builder object initialized with overlap joins and constraints
            potentially included, with the remainder still present in
            `QueryJoinPlans.predicate`.
        """
        # The fact that this method returns both a QueryPlan and an initial
        # QueryBuilder (rather than just a QueryPlan) is a tradeoff that lets
        # DimensionRecordStorageManager.process_query_overlaps (which is called
        # by the `_analyze_query_tree` call below) pull out overlap expressions
        # from the predicate at the same time it turns them into SQL table
        # joins (in the builder).
        joins_plan, builder = self._analyze_query_tree(tree)

        # The "projection" columns differ from the final columns by not
        # omitting any dimension keys (this keeps queries for different result
        # types more similar during construction), including any columns needed
        # only by order_by terms, and including the collection key if we need
        # it for GROUP BY or DISTINCT.
        projection_plan = QueryProjectionPlan(
            final_columns.copy(), joins_plan.datasets, find_first_dataset=find_first_dataset
        )
        projection_plan.columns.restore_dimension_keys()
        for term in order_by:
            term.gather_required_columns(projection_plan.columns)
        # The projection gets interesting if it does not have all of the
        # dimension keys or dataset fields of the "joins" stage, because that
        # means it needs to do a GROUP BY or DISTINCT ON to get unique rows.
        if projection_plan.columns.dimensions != joins_plan.columns.dimensions:
            assert projection_plan.columns.dimensions.issubset(joins_plan.columns.dimensions)
            # We're going from a larger set of dimensions to a smaller set,
            # that means we'll be doing a SELECT DISTINCT [ON] or GROUP BY.
            projection_plan.needs_dimension_distinct = True
        for dataset_type, fields_for_dataset in joins_plan.columns.dataset_fields.items():
            if not projection_plan.columns.dataset_fields[dataset_type]:
                # The "joins"-stage query has one row for each collection for
                # each data ID, but the projection-stage query just wants
                # one row for each data ID.
                if len(joins_plan.datasets[dataset_type].collection_records) > 1:
                    projection_plan.needs_dataset_distinct = True
                    break
        # If there are any dataset fields being propagated through that
        # projection and there is more than one collection, we need to
        # include the collection_key column so we can use that as one of
        # the DISTINCT or GROUP BY columns.
        for dataset_type, fields_for_dataset in projection_plan.columns.dataset_fields.items():
            if len(joins_plan.datasets[dataset_type].collection_records) > 1:
                fields_for_dataset.add("collection_key")
        if projection_plan:
            # If there's a projection and we're doing postprocessing, we might
            # be collapsing the dimensions of the postprocessing regions.  When
            # that happens, we want to apply an aggregate function to them that
            # computes the union of the regions that are grouped together.
            for element in builder.postprocessing.iter_missing(projection_plan.columns):
                if element.name not in projection_plan.columns.dimensions.elements:
                    projection_plan.region_aggregates.append(element)

        # The joins-stage query also needs to include all columns needed by the
        # downstream projection query.  Note that this:
        # - never adds new dimensions to the joins stage (since those are
        #   always a superset of the projection-stage dimensions);
        # - does not affect our determination of
        #   projection_plan.needs_dataset_distinct, because any dataset fields
        #   being added to the joins stage here are already in the projection.
        joins_plan.columns.update(projection_plan.columns)

        find_first_plan = None
        if find_first_dataset is not None:
            find_first_plan = QueryFindFirstPlan(joins_plan.datasets[find_first_dataset])
            # If we're doing a find-first search and there's a calibration
            # collection in play, we need to make sure the rows coming out of
            # the base query have only one timespan for each data ID +
            # collection, and we can only do that with a GROUP BY and COUNT
            # that we inspect in postprocessing.
            if find_first_plan.search.is_calibration_search:
                builder.postprocessing.check_validity_match_count = True
        plan = QueryPlan(
            joins=joins_plan,
            projection=projection_plan,
            find_first=find_first_plan,
            final_columns=final_columns,
        )
        return plan, builder

    def apply_query_joins(self, plan: QueryJoinsPlan, joiner: QueryJoiner) -> None:
        """Modify a `QueryJoiner` to include all tables and other FROM and
        WHERE clause terms needed.

        Parameters
        ----------
        plan : `QueryJoinPlan`
            Component of a `QueryPlan` relevant for the "joins" stage.
        joiner : `QueryJoiner`
            Component of a `QueryBuilder` that holds the FROM and WHERE
            clauses.  This is expected to be initialized by `analyze_query`
            and will be modified in-place on return.
        """
        # Process data coordinate upload joins.
        for upload_key, upload_dimensions in plan.data_coordinate_uploads.items():
            joiner.join(
                QueryJoiner(self.db, self._upload_tables[upload_key]).extract_dimensions(
                    upload_dimensions.required
                )
            )
        # Process materialization joins. We maintain a set of dataset types
        # that were included in a materialization; searches for these datasets
        # can be dropped if they are only present to provide a constraint on
        # data IDs, since that's already embedded in a materialization.
        materialized_datasets: set[str] = set()
        for materialization_key, materialization_dimensions in plan.materializations.items():
            materialized_datasets.update(
                self._join_materialization(joiner, materialization_key, materialization_dimensions)
            )
        # Process dataset joins.
        for dataset_search in plan.datasets.values():
            self._join_dataset_search(
                joiner,
                dataset_search,
                plan.columns.dataset_fields[dataset_search.name],
            )
        # Join in dimension element tables that we know we need relationships
        # or columns from.
        for element in plan.iter_mandatory():
            joiner.join(
                self.managers.dimensions.make_query_joiner(
                    element, plan.columns.dimension_fields[element.name]
                )
            )
        # See if any dimension keys are still missing, and if so join in their
        # tables.  Note that we know there are no fields needed from these.
        while not (joiner.dimension_keys.keys() >= plan.columns.dimensions.names):
            # Look for opportunities to join in multiple dimensions via single
            # table, to reduce the total number of tables joined in.
            missing_dimension_names = plan.columns.dimensions.names - joiner.dimension_keys.keys()
            best = self._universe[
                max(
                    missing_dimension_names,
                    key=lambda name: len(self._universe[name].dimensions.names & missing_dimension_names),
                )
            ]
            joiner.join(self.managers.dimensions.make_query_joiner(best, frozenset()))
        # Add the WHERE clause to the joiner.
        joiner.where(plan.predicate.visit(SqlColumnVisitor(joiner, self)))

    def apply_query_projection(self, plan: QueryProjectionPlan, builder: QueryBuilder) -> None:
        """Modify `QueryBuilder` to reflect the "projection" stage of query
        construction, which can involve a GROUP BY or DISTINCT [ON] clause
        that enforces uniqueness.

        Parameters
        ----------
        plan : `QueryProjectionPlan`
            Component of a `QueryPlan` relevant for the "projection" stage.
        builder : `QueryBuilder`
            Builder object that will be modified in place.  Expected to be
            initialized by `analyze_query` and further modified by
            `apply_query_joins`.
        """
        builder.columns = plan.columns
        if not plan and not builder.postprocessing.check_validity_match_count:
            # Rows are already unique; nothing else to do in this method.
            return
        # This method generates  either a SELECT DISTINCT [ON] or a SELECT with
        # GROUP BY. We'll work out which as we go.
        have_aggregates: bool = False
        # Dimension key columns form at least most of our GROUP BY or DISTINCT
        # ON clause.
        unique_keys: list[sqlalchemy.ColumnElement[Any]] = [
            builder.joiner.dimension_keys[k][0] for k in plan.columns.dimensions.data_coordinate_keys
        ]
        # There are two reasons we might need an aggregate function:
        # - to make sure temporal constraints and joins have resulted in at
        #   most one validity range match for each data ID and collection,
        #   when we're doing a find-first query.
        # - to compute the unions of regions we need for postprocessing, when
        #   the data IDs for those regions are not wholly included in the
        #   results (i.e. we need to postprocess on
        #   visit_detector_region.region, but the output rows don't have
        #   detector, just visit - so we compute the union of the
        #   visit_detector region over all matched detectors).
        if builder.postprocessing.check_validity_match_count:
            builder.joiner.special[builder.postprocessing.VALIDITY_MATCH_COUNT] = (
                sqlalchemy.func.count().label(builder.postprocessing.VALIDITY_MATCH_COUNT)
            )
            have_aggregates = True
        for element in plan.region_aggregates:
            builder.joiner.fields[element.name]["region"] = ddl.Base64Region.union_aggregate(
                builder.joiner.fields[element.name]["region"]
            )
            have_aggregates = True
        # Many of our fields derive their uniqueness from the unique_key
        # fields: if rows are uniqe over the 'unique_key' fields, then they're
        # automatically unique over these 'derived_fields'.  We just remember
        # these as pairs of (logical_table, field) for now.
        derived_fields: list[tuple[str, str]] = []
        # All dimension record fields are derived fields.
        for element_name, fields_for_element in plan.columns.dimension_fields.items():
            for element_field in fields_for_element:
                derived_fields.append((element_name, element_field))
        # Some dataset fields are derived fields and some are unique keys, and
        # it depends on the kinds of collection(s) we're searching and whether
        # it's a find-first query.
        for dataset_type, fields_for_dataset in plan.columns.dataset_fields.items():
            for dataset_field in fields_for_dataset:
                if dataset_field == "collection_key":
                    # If the collection_key field is present, it's needed for
                    # uniqueness if we're looking in more than one collection.
                    # If not, it's a derived field.
                    if len(plan.datasets[dataset_type].collection_records) > 1:
                        unique_keys.append(builder.joiner.fields[dataset_type]["collection_key"])
                    else:
                        derived_fields.append((dataset_type, "collection_key"))
                elif dataset_field == "timespan" and plan.datasets[dataset_type].is_calibration_search:
                    # If we're doing a non-find-first query against a
                    # CALIBRATION collection, the timespan is also a unique
                    # key...
                    if dataset_type == plan.find_first_dataset:
                        # ...unless we're doing a find-first search on this
                        # dataset, in which case we need to use ANY_VALUE on
                        # the timespan and check that _VALIDITY_MATCH_COUNT
                        # (added earlier) is one, indicating that there was
                        # indeed only one timespan for each data ID in each
                        # collection that survived the base query's WHERE
                        # clauses and JOINs.
                        if not self.db.has_any_aggregate:
                            raise NotImplementedError(
                                f"Cannot generate query that returns {dataset_type}.timespan after a "
                                "find-first search, because this a database does not support the ANY_VALUE "
                                "aggregate function (or equivalent)."
                            )
                        builder.joiner.timespans[dataset_type] = builder.joiner.timespans[
                            dataset_type
                        ].apply_any_aggregate(self.db.apply_any_aggregate)
                    else:
                        unique_keys.extend(builder.joiner.timespans[dataset_type].flatten())
                else:
                    # Other dataset fields derive their uniqueness from key
                    # fields.
                    derived_fields.append((dataset_type, dataset_field))
        if not have_aggregates and not derived_fields:
            # SELECT DISTINCT is sufficient.
            builder.distinct = True
        elif not have_aggregates and self.db.has_distinct_on:
            # SELECT DISTINCT ON is sufficient and supported by this database.
            builder.distinct = unique_keys
        else:
            # GROUP BY is the only option.
            if derived_fields:
                if self.db.has_any_aggregate:
                    for logical_table, field in derived_fields:
                        if field == "timespan":
                            builder.joiner.timespans[logical_table] = builder.joiner.timespans[
                                logical_table
                            ].apply_any_aggregate(self.db.apply_any_aggregate)
                        else:
                            builder.joiner.fields[logical_table][field] = self.db.apply_any_aggregate(
                                builder.joiner.fields[logical_table][field]
                            )
                else:
                    _LOG.warning(
                        "Adding %d fields to GROUP BY because this database backend does not support the "
                        "ANY_VALUE aggregate function (or equivalent).  This may result in a poor query "
                        "plan.  Materializing the query first sometimes avoids this problem.",
                        len(derived_fields),
                    )
                    for logical_table, field in derived_fields:
                        if field == "timespan":
                            unique_keys.extend(builder.joiner.timespans[logical_table].flatten())
                        else:
                            unique_keys.append(builder.joiner.fields[logical_table][field])
            builder.group_by = unique_keys

    def apply_query_find_first(self, plan: QueryFindFirstPlan | None, builder: QueryBuilder) -> QueryBuilder:
        """Modify an under-construction SQL query to return only one row for
        each data ID, searching collections in order.

        Parameters
        ----------
        plan : `QueryFindFirstPlan` or `None`
            Component of a `QueryPlan` relevant for the "find first" stage.
        builder : `QueryBuilder`
            Builder object as produced by `apply_query_projection`.  This
            object should be considered to be consumed by this method - the
            same instance may or may not be returned, and if it is not
            returned, its state is not defined.

        Returns
        -------
        builder : `QueryBuilder`
            Modified query builder that includes the find-first resolution, if
            one was needed.
        """
        if not plan:
            return builder
        # The query we're building looks like this:
        #
        # WITH {dst}_base AS (
        #     {target}
        #     ...
        # )
        # SELECT
        #     {dst}_window.*,
        # FROM (
        #     SELECT
        #         {dst}_base.*,
        #         ROW_NUMBER() OVER (
        #             PARTITION BY {dst_base}.{dimensions}
        #             ORDER BY {rank}
        #         ) AS rownum
        #     ) {dst}_window
        # WHERE
        #     {dst}_window.rownum = 1;
        #
        # The outermost SELECT will be represented by the QueryBuilder we
        # return. The QueryBuilder we're given corresponds to the Common Table
        # Expression (CTE) at the top.
        #
        # For SQLite only, we could use a much simpler GROUP BY instead,
        # because it extends the standard to do exactly what we want when MIN
        # or MAX appears once and a column does not have an aggregate function
        # (https://www.sqlite.org/quirks.html).  But since that doesn't work
        # with PostgreSQL it doesn't help us.
        #
        builder = builder.nested(cte=True, force=True)
        # We start by filling out the "window" SELECT statement...
        partition_by = [builder.joiner.dimension_keys[d][0] for d in builder.columns.dimensions.required]
        rank_sql_column = sqlalchemy.case(
            {record.key: n for n, record in enumerate(plan.search.collection_records)},
            value=builder.joiner.fields[plan.dataset_type]["collection_key"],
        )
        if partition_by:
            builder.joiner.special["_ROWNUM"] = sqlalchemy.sql.func.row_number().over(
                partition_by=partition_by, order_by=rank_sql_column
            )
        else:
            builder.joiner.special["_ROWNUM"] = sqlalchemy.sql.func.row_number().over(
                order_by=rank_sql_column
            )
        # ... and then turn that into a subquery with a constraint on rownum.
        builder = builder.nested(force=True)
        # We can now add the WHERE constraint on rownum into the outer query.
        builder.joiner.where(builder.joiner.special["_ROWNUM"] == 1)
        # Don't propagate _ROWNUM into downstream queries.
        del builder.joiner.special["_ROWNUM"]
        return builder

    def _analyze_query_tree(self, tree: qt.QueryTree) -> tuple[QueryJoinsPlan, QueryBuilder]:
        """Start constructing a plan for building a query from a
        `.queries.tree.QueryTree`.

        Parameters
        ----------
        tree : `.queries.tree.QueryTree`
            Description of the joins and row filters in the query.

        Returns
        -------
        plan : `QueryJoinsPlan`
            Initial component of the plan relevant for the "joins" stage,
            including all joins and columns needed by ``tree``.  Additional
            columns will be added to this plan later.
        builder : `QueryBuilder`
            Builder object initialized with overlap joins and constraints
            potentially included, with the remainder still present in
            `QueryJoinPlans.predicate`.
        """
        # Delegate to the dimensions manager to rewrite the predicate and start
        # a QueryBuilder to cover any spatial overlap joins or constraints.
        # We'll return that QueryBuilder at the end.
        (
            predicate,
            builder,
        ) = self.managers.dimensions.process_query_overlaps(
            tree.dimensions,
            tree.predicate,
            tree.get_joined_dimension_groups(),
        )
        result = QueryJoinsPlan(predicate=predicate, columns=builder.columns)
        # Add columns required by postprocessing.
        builder.postprocessing.gather_columns_required(result.columns)
        # We also check that the predicate doesn't reference any dimensions
        # without constraining their governor dimensions, since that's a
        # particularly easy mistake to make and it's almost never intentional.
        # We also allow the registry data ID values to provide governor values.
        where_columns = qt.ColumnSet(self.universe.empty.as_group())
        result.predicate.gather_required_columns(where_columns)
        for governor in where_columns.dimensions.governors:
            if governor not in result.constraint_data_id:
                if governor in self._defaults.dataId.dimensions:
                    result.constraint_data_id[governor] = self._defaults.dataId[governor]
                else:
                    raise InvalidQueryError(
                        f"Query 'where' expression references a dimension dependent on {governor} without "
                        "constraining it directly."
                    )
        # Add materializations, which can also bring in more postprocessing.
        for m_key, m_dimensions in tree.materializations.items():
            m_state = self._materializations[m_key]
            result.materializations[m_key] = m_dimensions
            # When a query is materialized, the new tree has an empty
            # (trivially true) predicate because the original was used to make
            # the materialized rows.  But the original postprocessing isn't
            # executed when the materialization happens, so we have to include
            # it here.
            builder.postprocessing.spatial_join_filtering.extend(
                m_state.postprocessing.spatial_join_filtering
            )
            builder.postprocessing.spatial_where_filtering.extend(
                m_state.postprocessing.spatial_where_filtering
            )
        # Add data coordinate uploads.
        result.data_coordinate_uploads.update(tree.data_coordinate_uploads)
        # Add dataset_searches and filter out collections that don't have the
        # right dataset type or governor dimensions.
        for dataset_type_name, dataset_search in tree.datasets.items():
            resolved_dataset_search = self._resolve_dataset_search(
                dataset_type_name, dataset_search, result.constraint_data_id
            )
            result.datasets[dataset_type_name] = resolved_dataset_search
            if not resolved_dataset_search.collection_records:
                result.messages.append(f"Search for dataset type {dataset_type_name!r} is doomed to fail.")
                result.messages.extend(resolved_dataset_search.messages)
        return result, builder

    def _resolve_dataset_search(
        self,
        dataset_type_name: str,
        dataset_search: qt.DatasetSearch,
        constraint_data_id: Mapping[str, DataIdValue],
    ) -> ResolvedDatasetSearch:
        """Resolve the collections that should actually be searched for
        datasets of a particular type.

        Parameters
        ----------
        dataset_type_name : `str`
            Name of the dataset being searched for.
        dataset_search : `.queries.tree.DatasetSearch`
            Struct holding the dimensions and original collection search path.
        constraint_data_id : `~collections.abc.Mapping`
            Data ID mapping derived from the query predicate that may be used
            to filter out some collections based on their governor dimensions.

        Returns
        -------
        resolved : `ResolvedDatasetSearch`
            Struct that extends `dataset_search`` with the dataset type name
            and resolved collection records.
        """
        result = ResolvedDatasetSearch(dataset_type_name, dataset_search.dimensions)
        for collection_record, collection_summary in self._resolve_collection_path(
            dataset_search.collections
        ):
            rejected: bool = False
            if result.name not in collection_summary.dataset_types.names:
                result.messages.append(
                    f"No datasets of type {result.name!r} in collection {collection_record.name!r}."
                )
                rejected = True
            for governor in constraint_data_id.keys() & collection_summary.governors.keys():
                if constraint_data_id[governor] not in collection_summary.governors[governor]:
                    result.messages.append(
                        f"No datasets with {governor}={constraint_data_id[governor]!r} "
                        f"in collection {collection_record.name!r}."
                    )
                    rejected = True
            if not rejected:
                if collection_record.type is CollectionType.CALIBRATION:
                    result.is_calibration_search = True
                result.collection_records.append(collection_record)
        if result.dimensions != self.get_dataset_type(dataset_type_name).dimensions.as_group():
            # This is really for server-side defensiveness; it's hard to
            # imagine the query getting different dimensions for a dataset
            # type in two calls to the same query driver.
            raise InvalidQueryError(
                f"Incorrect dimensions {result.dimensions} for dataset {dataset_type_name} "
                f"in query (vs. {self.get_dataset_type(dataset_type_name).dimensions.as_group()})."
            )
        return result

    def _resolve_collection_path(
        self, collections: Iterable[str]
    ) -> list[tuple[CollectionRecord, CollectionSummary]]:
        """Expand an ordered iterable of collection names into a list of
        collection records and summaries.

        Parameters
        ----------
        collections : `~collections.abc.Iterable` [ `str` ]
            Ordered iterable of collections.

        Returns
        -------
        resolved : `list` [ `tuple` [ `.registry.interfaces.CollectionRecord`,\
                `.registry.CollectionSummary` ] ]
            Tuples of collection record and summary.  `~CollectionType.CHAINED`
            collections are flattened out and not included.
        """
        result: list[tuple[CollectionRecord, CollectionSummary]] = []
        done: set[str] = set()

        # Eventually we really want this recursive Python code to be replaced
        # by a recursive SQL query, especially if we extend this method to
        # support collection glob patterns to support public APIs we don't yet
        # have in the new query system (but will need to add).

        def recurse(collection_names: Iterable[str]) -> None:
            for collection_name in collection_names:
                if collection_name not in done:
                    done.add(collection_name)
                    record = self.managers.collections.find(collection_name)

                    if record.type is CollectionType.CHAINED:
                        recurse(cast(ChainedCollectionRecord, record).children)
                    else:
                        result.append((record, self.managers.datasets.getCollectionSummary(record)))

        recurse(collections)

        return result

    def _join_materialization(
        self,
        joiner: QueryJoiner,
        key: qt.MaterializationKey,
        dimensions: DimensionGroup,
    ) -> frozenset[str]:
        """Join a materialization into an under-construction query.

        Parameters
        ----------
        joiner : `QueryJoiner`
            Component of a `QueryBuilder` that holds the FROM and WHERE
            clauses.  This will be modified in-place on return.
        key : `.queries.tree.MaterializationKey`
            Unique identifier created for this materialization when it was
            created.
        dimensions : `DimensionGroup`
            Dimensions of the materialization.

        Returns
        -------
        datasets : `frozenset` [ `str` ]
            Dataset types that were included as constraints when this
            materialization was created.
        """
        columns = qt.ColumnSet(dimensions)
        m_state = self._materializations[key]
        joiner.join(QueryJoiner(self.db, m_state.table).extract_columns(columns, m_state.postprocessing))
        return m_state.datasets

    def _join_dataset_search(
        self,
        joiner: QueryJoiner,
        resolved_search: ResolvedDatasetSearch,
        fields: Set[str],
    ) -> None:
        """Join a dataset search into an under-construction query.

        Parameters
        ----------
        joiner : `QueryJoiner`
            Component of a `QueryBuilder` that holds the FROM and WHERE
            clauses.  This will be modified in-place on return.
        resolved_search : `ResolvedDatasetSearch`
            Struct that describes the dataset type and collections.
        fields : `~collections.abc.Set` [ `str` ]
            Dataset fields to include.
        """
        storage = self.managers.datasets[resolved_search.name]
        # The next two asserts will need to be dropped (and the implications
        # dealt with instead) if materializations start having dataset fields.
        assert (
            resolved_search.name not in joiner.fields
        ), "Dataset fields have unexpectedly already been joined in."
        assert (
            resolved_search.name not in joiner.timespans
        ), "Dataset timespan has unexpectedly already been joined in."
        joiner.join(storage.make_query_joiner(resolved_search.collection_records, fields))


@dataclasses.dataclass
class _MaterializationState:
    table: sqlalchemy.Table
    datasets: frozenset[str]
    postprocessing: Postprocessing


class _Cursor:
    """A helper class for managing paged query results and cursor lifetimes.

    This class holds a context manager for the SQLAlchemy cursor object but is
    not itself a context manager.  It always cleans up (i.e. calls its `close`
    method) when it raises an exception or exhausts the cursor, but external
    code is responsible for calling `close` when the cursor is abandoned before
    it is exhausted, including when that happens due to an external exception.

    Parameters
    ----------
    db : `.registry.interface.Database`
        Database to run the query against.
    sql : `sqlalchemy.Executable`
        SQL query to execute.
    result : `ResultSpec`
        Specification of the result type.
    name_shrinker : `NameShrinker` or `None`
        Object that was used to shrink dataset column names to fit within the
        database identifier limit.
    postprocessing : `Postprocessing`
        Post-query filtering and checks to perform.
    raw_page_size : `int`
        Maximum number of SQL result rows to return in each page, before
        postprocessing.
    """

    def __init__(
        self,
        db: Database,
        sql: sqlalchemy.Executable,
        result_spec: ResultSpec,
        name_shrinker: NameShrinker | None,
        postprocessing: Postprocessing,
        raw_page_size: int,
    ):
        self._result_spec = result_spec
        self._name_shrinker = name_shrinker
        self._raw_page_size = raw_page_size
        self._postprocessing = postprocessing
        self._timespan_repr_cls = db.getTimespanRepresentation()
        self._context = db.query(sql, execution_options=dict(yield_per=raw_page_size))
        cursor = self._context.__enter__()
        try:
            self._iterator = cursor.partitions()
        except:  # noqa: E722
            self._context.__exit__(*sys.exc_info())
            raise

    def close(self, exc_type: Any = None, exc_value: Any = None, traceback: Any = None) -> None:
        """Close this cursor.

        Parameters
        ----------
        exc_type : `type`
            Exception type as obtained from `sys.exc_info`, or `None` if there
            was no error.
        exc_value : `BaseException` or `None`
            Exception instance as obtained from `sys.exc_info`, or `None` if
            there was no error.
        traceback : `object`
            Traceback as obtained from `sys.exc_info`, or `None` if there was
            no error.
        """
        self._context.__exit__(exc_type, exc_value, traceback)

    def next(self) -> ResultPage:
        """Return the next result page from this query.

        When there are no more results after this result page, the `next_page`
        attribute of the returned object is `None` and the cursor will be
        closed.  The cursor is also closed if this method raises an exception.
        """
        try:
            raw_page = next(self._iterator, tuple())
            if len(raw_page) == self._raw_page_size:
                # There's some chance we got unlucky and this page exactly
                # finishes off the query, and we won't know the next page does
                # not exist until we try to fetch it.  But that's better than
                # always fetching the next page up front.
                next_key = uuid.uuid4()
            else:
                next_key = None
                self.close()

            postprocessed_rows = self._postprocessing.apply(raw_page)
            match self._result_spec:
                case DimensionRecordResultSpec():
                    return self._convert_dimension_record_results(postprocessed_rows, next_key)
                case _:
                    raise NotImplementedError("TODO")
        except:  # noqa: E722
            self._context.__exit__(*sys.exc_info())
            raise

    def _convert_dimension_record_results(
        self,
        raw_rows: Iterable[sqlalchemy.Row],
        next_key: PageKey | None,
    ) -> DimensionRecordResultPage:
        """Convert a raw SQL result iterable into a page of `DimensionRecord`
        query results.

        Parameters
        ----------
        raw_rows : `~collections.abc.Iterable` [ `sqlalchemy.Row` ]
            Iterable of SQLAlchemy rows, with `Postprocessing` filters already
            applied.
        next_key : `PageKey` or `None`
            Key for the next page to add into the returned page object.

        Returns
        -------
        result_page : `DimensionRecordResultPage`
            Page object that holds a `DimensionRecord` container.
        """
        result_spec = cast(DimensionRecordResultSpec, self._result_spec)
        record_set = DimensionRecordSet(result_spec.element)
        record_cls = result_spec.element.RecordClass
        if isinstance(result_spec.element, SkyPixDimension):
            pixelization = result_spec.element.pixelization
            id_qualified_name = qt.ColumnSet.get_qualified_name(result_spec.element.name, None)
            for raw_row in raw_rows:
                pixel_id = raw_row._mapping[id_qualified_name]
                record_set.add(record_cls(id=pixel_id, region=pixelization.pixel(pixel_id)))
        else:
            # Mapping from DimensionRecord attribute name to qualified column
            # name, but as a list of tuples since we'd just iterate over items
            # anyway.
            column_map = list(
                zip(
                    result_spec.element.schema.dimensions.names,
                    result_spec.element.dimensions.names,
                )
            )
            for field in result_spec.element.schema.remainder.names:
                if field != "timespan":
                    column_map.append(
                        (field, qt.ColumnSet.get_qualified_name(result_spec.element.name, field))
                    )
            if result_spec.element.temporal:
                timespan_qualified_name = qt.ColumnSet.get_qualified_name(
                    result_spec.element.name, "timespan"
                )
            else:
                timespan_qualified_name = None
            for raw_row in raw_rows:
                m = raw_row._mapping
                d = {k: m[v] for k, v in column_map}
                if timespan_qualified_name is not None:
                    d["timespan"] = self._timespan_repr_cls.extract(m, name=timespan_qualified_name)
                record_set.add(record_cls(**d))
        return DimensionRecordResultPage(spec=result_spec, next_key=next_key, rows=record_set)
