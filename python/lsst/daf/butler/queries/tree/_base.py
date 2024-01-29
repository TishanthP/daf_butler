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

__all__ = (
    "QueryTreeBase",
    "ColumnExpressionBase",
    "DatasetFieldName",
    "InvalidQueryTreeError",
    "DATASET_FIELD_NAMES",
)

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, TypeVar, cast, get_args

import pydantic

if TYPE_CHECKING:
    from ...column_spec import ColumnType
    from ..visitors import ColumnExpressionVisitor, PredicateVisitFlags, PredicateVisitor
    from ._column_literal import ColumnLiteral
    from ._column_set import ColumnSet
    from ._predicate import PredicateLeaf


DatasetFieldName: TypeAlias = Literal["dataset_id", "ingest_date", "run", "collection", "timespan"]

DATASET_FIELD_NAMES: tuple[DatasetFieldName, ...] = tuple(get_args(DatasetFieldName))

_T = TypeVar("_T")
_L = TypeVar("_L")
_A = TypeVar("_A")
_O = TypeVar("_O")


class InvalidQueryTreeError(RuntimeError):
    """Exception raised when a query tree is or would not be valid."""


class QueryTreeBase(pydantic.BaseModel):
    """Base class for all non-primitive types in a query tree."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", strict=True)


class ColumnExpressionBase(QueryTreeBase, ABC):
    """Base class for objects that represent non-boolean column expressions in
    a query tree.

    Notes
    -----
    This is a closed hierarchy whose concrete, `~typing.final` derived classes
    are members of the `ColumnExpression` union.  That union should generally
    be used in type annotations rather than the technically-open base class.
    """

    expression_type: str

    is_literal: ClassVar[bool] = False
    """Whether this expression wraps a literal Python value."""

    @property
    @abstractmethod
    def precedence(self) -> int:
        """Operator precedence for this operation.

        Lower values bind more tightly, so parentheses are needed when printing
        an expression where an operand has a higher value than the expression
        itself.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def column_type(self) -> ColumnType:
        """A string enumeration value representing the type of the column
        expression.
        """
        raise NotImplementedError()

    def get_literal_value(self) -> Any | None:
        """Return the literal value wrapped by this expression, or `None` if
        it is not a literal.
        """
        return None

    @abstractmethod
    def gather_required_columns(self, columns: ColumnSet) -> None:
        """Add any columns required to evaluate this expression to the
        given column set.

        Parameters
        ----------
        columns : `ColumnSet`
            Set of columns to modify in place.
        """
        raise NotImplementedError()

    @abstractmethod
    def visit(self, visitor: ColumnExpressionVisitor[_T]) -> _T:
        """Invoke the visitor interface.

        Parameters
        ----------
        visitor : `ColumnExpressionVisitor`
            Visitor to invoke a method on.

        Returns
        -------
        result : `object`
            Forwarded result from the visitor.
        """
        raise NotImplementedError()


class ColumnLiteralBase(ColumnExpressionBase):
    """Base class for objects that represent literal values as column
    expressions in a query tree.

    Notes
    -----
    This is a closed hierarchy whose concrete, `~typing.final` derived classes
    are members of the `ColumnLiteral` union.  That union should generally be
    used in type annotations rather than the technically-open base class. The
    concrete members of that union are only semi-public; they appear in the
    serialized form of a column expression tree, but should only be constructed
    via the `make_column_literal` factory function.  All concrete members of
    the union are also guaranteed to have a read-only ``value`` attribute
    holding the wrapped literal, but it is unspecified whether that is a
    regular attribute or a `property`.
    """

    is_literal: ClassVar[bool] = True
    """Whether this expression wraps a literal Python value."""

    @property
    def precedence(self) -> int:
        # Docstring inherited.
        return 0

    def get_literal_value(self) -> Any:
        # Docstring inherited.
        return cast("ColumnLiteral", self).value

    def gather_required_columns(self, columns: ColumnSet) -> None:
        # Docstring inherited.
        pass

    @property
    def column_type(self) -> ColumnType:
        # Docstring inherited.
        return cast(ColumnType, self.expression_type)

    def visit(self, visitor: ColumnExpressionVisitor[_T]) -> _T:
        # Docstring inherited
        return visitor.visit_literal(cast("ColumnLiteral", self))


class PredicateLeafBase(QueryTreeBase, ABC):
    """Base class for leaf nodes of the `Predicate` tree.

    Notes
    -----
    This is a closed hierarchy whose concrete, `~typing.final` derived classes
    are members of the `PredicateLeaf` union.  That union should generally be
    used in type annotations rather than the technically-open base class. The
    concrete members of that union are only semi-public; they appear in the
    serialized form of a `Predicate`, but should only be constructed
    via various `Predicate` factory methods.
    """

    @property
    @abstractmethod
    def precedence(self) -> int:
        """Operator precedence for this operation.

        Lower values bind more tightly, so parentheses are needed when printing
        an expression where an operand has a higher value than the expression
        itself.
        """
        raise NotImplementedError()

    @property
    def column_type(self) -> Literal["bool"]:
        """A string enumeration value representing the type of the column
        expression.
        """
        return "bool"

    @abstractmethod
    def gather_required_columns(self, columns: ColumnSet) -> None:
        """Add any columns required to evaluate this predicate leaf to the
        given column set.

        Parameters
        ----------
        columns : `ColumnSet`
            Set of columns to modify in place.
        """
        raise NotImplementedError()

    def invert(self) -> PredicateLeaf:
        """Return a new leaf that is the logical not of this one."""
        from ._predicate import LogicalNot, LogicalNotOperand

        # This implementation works for every subclass other than LogicalNot
        # itself, which overrides this method.
        return LogicalNot.model_construct(operand=cast(LogicalNotOperand, self))

    @abstractmethod
    def visit(self, visitor: PredicateVisitor[_A, _O, _L], flags: PredicateVisitFlags) -> _L:
        """Invoke the visitor interface.

        Parameters
        ----------
        visitor : `PredicateVisitor`
            Visitor to invoke a method on.
        flags : `PredicateVisitFlags`
            Flags that provide information about where this leaf appears in the
            larger predicate tree.

        Returns
        -------
        result : `object`
            Forwarded result from the visitor.
        """
        raise NotImplementedError()
