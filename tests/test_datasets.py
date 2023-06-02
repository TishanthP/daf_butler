# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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

import copy
import pickle
import unittest
import uuid

from lsst.daf.butler import (
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionUniverse,
    FileDataset,
    StorageClass,
    StorageClassFactory,
)

"""Tests for datasets module.
"""


class DatasetTypeTestCase(unittest.TestCase):
    """Test for DatasetType."""

    def setUp(self):
        self.universe = DimensionUniverse()

    def testConstructor(self):
        """Test construction preserves values.

        Note that construction doesn't check for valid storageClass.
        This can only be verified for a particular schema.
        """
        datasetTypeName = "test"
        storageClass = StorageClass("test_StructuredData")
        dimensions = self.universe.extract(("visit", "instrument"))
        datasetType = DatasetType(datasetTypeName, dimensions, storageClass)
        self.assertEqual(datasetType.name, datasetTypeName)
        self.assertEqual(datasetType.storageClass, storageClass)
        self.assertEqual(datasetType.dimensions, dimensions)

        with self.assertRaises(ValueError, msg="Construct component without parent storage class"):
            DatasetType(DatasetType.nameWithComponent(datasetTypeName, "comp"), dimensions, storageClass)
        with self.assertRaises(ValueError, msg="Construct non-component with parent storage class"):
            DatasetType(datasetTypeName, dimensions, storageClass, parentStorageClass="NotAllowed")

    def testConstructor2(self):
        """Test construction from StorageClass name."""
        datasetTypeName = "test"
        storageClass = StorageClass("test_constructor2")
        StorageClassFactory().registerStorageClass(storageClass)
        dimensions = self.universe.extract(("instrument", "visit"))
        datasetType = DatasetType(datasetTypeName, dimensions, "test_constructor2")
        self.assertEqual(datasetType.name, datasetTypeName)
        self.assertEqual(datasetType.storageClass, storageClass)
        self.assertEqual(datasetType.dimensions, dimensions)

    def testNameValidation(self):
        """Test that dataset type names only contain certain characters
        in certain positions.
        """
        dimensions = self.universe.extract(("instrument", "visit"))
        goodNames = ("a", "A", "z1", "Z1", "a_1B", "A_1b", "_a")
        badNames = ("1", "a%b", "B+Z", "T[0]")

        # Construct storage class with all the good names included as
        # components so that we can test internal consistency
        storageClass = StorageClass(
            "test_StructuredData", components={n: StorageClass("component") for n in goodNames}
        )

        for name in goodNames:
            composite = DatasetType(name, dimensions, storageClass)
            self.assertEqual(composite.name, name)
            for suffix in goodNames:
                full = DatasetType.nameWithComponent(name, suffix)
                component = composite.makeComponentDatasetType(suffix)
                self.assertEqual(component.name, full)
                self.assertEqual(component.parentStorageClass.name, "test_StructuredData")
            for suffix in badNames:
                full = DatasetType.nameWithComponent(name, suffix)
                with self.subTest(full=full):
                    with self.assertRaises(ValueError):
                        DatasetType(full, dimensions, storageClass)
        for name in badNames:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    DatasetType(name, dimensions, storageClass)

    def testEquality(self):
        storageA = StorageClass("test_a")
        storageB = StorageClass("test_b")
        parent = StorageClass("test")
        dimensionsA = self.universe.extract(["instrument"])
        dimensionsB = self.universe.extract(["skymap"])
        self.assertEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
        )
        self.assertEqual(
            DatasetType(
                "a",
                dimensionsA,
                "test_a",
            ),
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
        )
        self.assertEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsA,
                "test_a",
            ),
        )
        self.assertEqual(
            DatasetType(
                "a",
                dimensionsA,
                "test_a",
            ),
            DatasetType(
                "a",
                dimensionsA,
                "test_a",
            ),
        )
        self.assertEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass=parent),
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass=parent),
        )
        self.assertEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent"),
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent"),
        )
        self.assertNotEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent", isCalibration=True),
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent", isCalibration=False),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "b",
                dimensionsA,
                storageA,
            ),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "b",
                dimensionsA,
                "test_a",
            ),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsA,
                storageB,
            ),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsA,
                "test_b",
            ),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsB,
                storageA,
            ),
        )
        self.assertNotEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType(
                "a",
                dimensionsB,
                "test_a",
            ),
        )
        self.assertNotEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass=storageA),
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass=storageB),
        )
        self.assertNotEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="storageA"),
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="storageB"),
        )

    def testCompatibility(self):
        storageA = StorageClass("test_a", pytype=set, converters={"list": "builtins.set"})
        storageB = StorageClass("test_b", pytype=list)
        storageC = StorageClass("test_c", pytype=dict)
        self.assertTrue(storageA.can_convert(storageB))
        dimensionsA = self.universe.extract(["instrument"])

        dA = DatasetType("a", dimensionsA, storageA)
        dA2 = DatasetType("a", dimensionsA, storageB)
        self.assertNotEqual(dA, dA2)
        self.assertTrue(dA.is_compatible_with(dA))
        self.assertTrue(dA.is_compatible_with(dA2))
        self.assertFalse(dA2.is_compatible_with(dA))

        dA3 = DatasetType("a", dimensionsA, storageC)
        self.assertFalse(dA.is_compatible_with(dA3))

    def testOverrideStorageClass(self):
        storageA = StorageClass("test_a", pytype=list, converters={"dict": "builtins.list"})
        storageB = StorageClass("test_b", pytype=dict)
        dimensions = self.universe.extract(["instrument"])

        dA = DatasetType("a", dimensions, storageA)
        dB = dA.overrideStorageClass(storageB)
        self.assertNotEqual(dA, dB)
        self.assertEqual(dB.storageClass, storageB)

        round_trip = dB.overrideStorageClass(storageA)
        self.assertEqual(round_trip, dA)

        # Check that parents move over.
        parent = StorageClass("composite", components={"a": storageA, "c": storageA})
        dP = DatasetType("comp", dimensions, parent)
        dP_A = dP.makeComponentDatasetType("a")
        print(dP_A)
        dp_B = dP_A.overrideStorageClass(storageB)
        self.assertEqual(dp_B.storageClass, storageB)
        self.assertEqual(dp_B.parentStorageClass, parent)

    def testJson(self):
        storageA = StorageClass("test_a")
        dimensionsA = self.universe.extract(["instrument"])
        self.assertEqual(
            DatasetType(
                "a",
                dimensionsA,
                storageA,
            ),
            DatasetType.from_json(
                DatasetType(
                    "a",
                    dimensionsA,
                    storageA,
                ).to_json(),
                self.universe,
            ),
        )
        self.assertEqual(
            DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent"),
            DatasetType.from_json(
                DatasetType("a.b", dimensionsA, "test_b", parentStorageClass="parent").to_json(),
                self.universe,
            ),
        )

    def testSorting(self):
        """Can we sort a DatasetType"""
        storage = StorageClass("test_a")
        dimensions = self.universe.extract(["instrument"])

        d_a = DatasetType("a", dimensions, storage)
        d_f = DatasetType("f", dimensions, storage)
        d_p = DatasetType("p", dimensions, storage)

        sort = sorted([d_p, d_f, d_a])
        self.assertEqual(sort, [d_a, d_f, d_p])

        # Now with strings
        with self.assertRaises(TypeError):
            sort = sorted(["z", d_p, "c", d_f, d_a, "d"])

    def testHashability(self):
        """Test `DatasetType.__hash__`.

        This test is performed by checking that `DatasetType` entries can
        be inserted into a `set` and that unique values of its
        (`name`, `storageClass`, `dimensions`) parameters result in separate
        entries (and equal ones don't).

        This does not check for uniformity of hashing or the actual values
        of the hash function.
        """
        types = []
        unique = 0
        storageC = StorageClass("test_c")
        storageD = StorageClass("test_d")
        for name in ["a", "b"]:
            for storageClass in [storageC, storageD]:
                for dimensions in [("instrument",), ("skymap",)]:
                    datasetType = DatasetType(name, self.universe.extract(dimensions), storageClass)
                    datasetTypeCopy = DatasetType(name, self.universe.extract(dimensions), storageClass)
                    types.extend((datasetType, datasetTypeCopy))
                    unique += 1  # datasetType should always equal its copy
        self.assertEqual(len(set(types)), unique)  # all other combinations are unique

        # also check that hashes of instances constructed with StorageClass
        # name matches hashes of instances constructed with instances
        dimensions = self.universe.extract(["instrument"])
        self.assertEqual(
            hash(DatasetType("a", dimensions, storageC)), hash(DatasetType("a", dimensions, "test_c"))
        )
        self.assertEqual(
            hash(DatasetType("a", dimensions, "test_c")), hash(DatasetType("a", dimensions, "test_c"))
        )
        self.assertNotEqual(
            hash(DatasetType("a", dimensions, storageC)), hash(DatasetType("a", dimensions, "test_d"))
        )
        self.assertNotEqual(
            hash(DatasetType("a", dimensions, storageD)), hash(DatasetType("a", dimensions, "test_c"))
        )
        self.assertNotEqual(
            hash(DatasetType("a", dimensions, "test_c")), hash(DatasetType("a", dimensions, "test_d"))
        )

    def testDeepCopy(self):
        """Test that we can copy a dataset type."""
        storageClass = StorageClass("test_copy")
        datasetTypeName = "test"
        dimensions = self.universe.extract(("instrument", "visit"))
        datasetType = DatasetType(datasetTypeName, dimensions, storageClass)
        dcopy = copy.deepcopy(datasetType)
        self.assertEqual(dcopy, datasetType)

        # Now with calibration flag set
        datasetType = DatasetType(datasetTypeName, dimensions, storageClass, isCalibration=True)
        dcopy = copy.deepcopy(datasetType)
        self.assertEqual(dcopy, datasetType)
        self.assertTrue(dcopy.isCalibration())

        # And again with a composite
        componentStorageClass = StorageClass("copy_component")
        componentDatasetType = DatasetType(
            DatasetType.nameWithComponent(datasetTypeName, "comp"),
            dimensions,
            componentStorageClass,
            parentStorageClass=storageClass,
        )
        dcopy = copy.deepcopy(componentDatasetType)
        self.assertEqual(dcopy, componentDatasetType)

    def testPickle(self):
        """Test pickle support."""
        storageClass = StorageClass("test_pickle")
        datasetTypeName = "test"
        dimensions = self.universe.extract(("instrument", "visit"))
        # Un-pickling requires that storage class is registered with factory.
        StorageClassFactory().registerStorageClass(storageClass)
        datasetType = DatasetType(datasetTypeName, dimensions, storageClass)
        datasetTypeOut = pickle.loads(pickle.dumps(datasetType))
        self.assertIsInstance(datasetTypeOut, DatasetType)
        self.assertEqual(datasetType.name, datasetTypeOut.name)
        self.assertEqual(datasetType.dimensions.names, datasetTypeOut.dimensions.names)
        self.assertEqual(datasetType.storageClass, datasetTypeOut.storageClass)
        self.assertIsNone(datasetTypeOut.parentStorageClass)
        self.assertIs(datasetType.isCalibration(), datasetTypeOut.isCalibration())
        self.assertFalse(datasetTypeOut.isCalibration())

        datasetType = DatasetType(datasetTypeName, dimensions, storageClass, isCalibration=True)
        datasetTypeOut = pickle.loads(pickle.dumps(datasetType))
        self.assertIs(datasetType.isCalibration(), datasetTypeOut.isCalibration())
        self.assertTrue(datasetTypeOut.isCalibration())

        # And again with a composite
        componentStorageClass = StorageClass("pickle_component")
        StorageClassFactory().registerStorageClass(componentStorageClass)
        componentDatasetType = DatasetType(
            DatasetType.nameWithComponent(datasetTypeName, "comp"),
            dimensions,
            componentStorageClass,
            parentStorageClass=storageClass,
        )
        datasetTypeOut = pickle.loads(pickle.dumps(componentDatasetType))
        self.assertIsInstance(datasetTypeOut, DatasetType)
        self.assertEqual(componentDatasetType.name, datasetTypeOut.name)
        self.assertEqual(componentDatasetType.dimensions.names, datasetTypeOut.dimensions.names)
        self.assertEqual(componentDatasetType.storageClass, datasetTypeOut.storageClass)
        self.assertEqual(componentDatasetType.parentStorageClass, datasetTypeOut.parentStorageClass)
        self.assertEqual(datasetTypeOut.parentStorageClass.name, storageClass.name)
        self.assertEqual(datasetTypeOut, componentDatasetType)

        # Now with a string and not a real storage class to test that
        # pickling doesn't force the StorageClass to be resolved
        componentDatasetType = DatasetType(
            DatasetType.nameWithComponent(datasetTypeName, "comp"),
            dimensions,
            "StrangeComponent",
            parentStorageClass="UnknownParent",
        )
        datasetTypeOut = pickle.loads(pickle.dumps(componentDatasetType))
        self.assertEqual(datasetTypeOut, componentDatasetType)
        self.assertEqual(datasetTypeOut._parentStorageClassName, componentDatasetType._parentStorageClassName)

        # Now with a storage class that is created by the factory
        factoryStorageClassClass = StorageClassFactory.makeNewStorageClass("ParentClass")
        factoryComponentStorageClassClass = StorageClassFactory.makeNewStorageClass("ComponentClass")
        componentDatasetType = DatasetType(
            DatasetType.nameWithComponent(datasetTypeName, "comp"),
            dimensions,
            factoryComponentStorageClassClass(),
            parentStorageClass=factoryStorageClassClass(),
        )
        datasetTypeOut = pickle.loads(pickle.dumps(componentDatasetType))
        self.assertEqual(datasetTypeOut, componentDatasetType)
        self.assertEqual(datasetTypeOut._parentStorageClassName, componentDatasetType._parentStorageClassName)

    def test_composites(self):
        """Test components within composite DatasetTypes."""
        storageClassA = StorageClass("compA")
        storageClassB = StorageClass("compB")
        storageClass = StorageClass(
            "test_composite", components={"compA": storageClassA, "compB": storageClassB}
        )
        self.assertTrue(storageClass.isComposite())
        self.assertFalse(storageClassA.isComposite())
        self.assertFalse(storageClassB.isComposite())

        dimensions = self.universe.extract(("instrument", "visit"))

        datasetTypeComposite = DatasetType("composite", dimensions, storageClass)
        datasetTypeComponentA = datasetTypeComposite.makeComponentDatasetType("compA")
        datasetTypeComponentB = datasetTypeComposite.makeComponentDatasetType("compB")

        self.assertTrue(datasetTypeComposite.isComposite())
        self.assertFalse(datasetTypeComponentA.isComposite())
        self.assertTrue(datasetTypeComponentB.isComponent())
        self.assertFalse(datasetTypeComposite.isComponent())

        self.assertEqual(datasetTypeComposite.name, "composite")
        self.assertEqual(datasetTypeComponentA.name, "composite.compA")
        self.assertEqual(datasetTypeComponentB.component(), "compB")
        self.assertEqual(datasetTypeComposite.nameAndComponent(), ("composite", None))
        self.assertEqual(datasetTypeComponentA.nameAndComponent(), ("composite", "compA"))

        self.assertEqual(datasetTypeComponentA.parentStorageClass, storageClass)
        self.assertEqual(datasetTypeComponentB.parentStorageClass, storageClass)
        self.assertIsNone(datasetTypeComposite.parentStorageClass)

        with self.assertRaises(KeyError):
            datasetTypeComposite.makeComponentDatasetType("compF")


class DatasetRefTestCase(unittest.TestCase):
    """Test for DatasetRef."""

    def setUp(self):
        self.universe = DimensionUniverse()
        datasetTypeName = "test"
        self.componentStorageClass1 = StorageClass("Component1")
        self.componentStorageClass2 = StorageClass("Component2")
        self.parentStorageClass = StorageClass(
            "Parent", components={"a": self.componentStorageClass1, "b": self.componentStorageClass2}
        )
        dimensions = self.universe.extract(("instrument", "visit"))
        self.dataId = dict(instrument="DummyCam", visit=42)
        self.datasetType = DatasetType(datasetTypeName, dimensions, self.parentStorageClass)

    def testConstructor(self):
        """Test that construction preserves and validates values."""
        # Constructing a ref requires a run.
        with self.assertRaises(TypeError):
            DatasetRef(self.datasetType, self.dataId, id=uuid.uuid4())

        # Constructing an unresolved ref with run and/or components should
        # issue a ref with an id.
        run = "somerun"
        ref = DatasetRef(self.datasetType, self.dataId, run=run)
        self.assertEqual(ref.datasetType, self.datasetType)
        self.assertEqual(
            ref.dataId, DataCoordinate.standardize(self.dataId, universe=self.universe), msg=ref.dataId
        )
        self.assertIsNotNone(ref.id)

        # Passing a data ID that is missing dimensions should fail.
        with self.assertRaises(KeyError):
            DatasetRef(self.datasetType, {"instrument": "DummyCam"}, run="run")
        # Constructing a resolved ref should preserve run as well as everything
        # else.
        id_ = uuid.uuid4()
        ref = DatasetRef(self.datasetType, self.dataId, id=id_, run=run)
        self.assertEqual(ref.datasetType, self.datasetType)
        self.assertEqual(
            ref.dataId, DataCoordinate.standardize(self.dataId, universe=self.universe), msg=ref.dataId
        )
        self.assertIsInstance(ref.dataId, DataCoordinate)
        self.assertEqual(ref.id, id_)
        self.assertEqual(ref.run, run)

        with self.assertRaises(ValueError):
            DatasetRef(self.datasetType, self.dataId, run=run, id_generation_mode=42)

    def testSorting(self):
        """Can we sort a DatasetRef"""
        # All refs have the same run.
        ref1 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=1), run="run")
        ref2 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=10), run="run")
        ref3 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=22), run="run")

        # Enable detailed diff report
        self.maxDiff = None

        # This will sort them on visit number
        sort = sorted([ref3, ref1, ref2])
        self.assertEqual(sort, [ref1, ref2, ref3], msg=f"Got order: {[r.dataId for r in sort]}")

        # Now include different runs.
        ref1 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=43), run="b")
        self.assertEqual(ref1.run, "b")
        ref4 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=10), run="b")
        ref2 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=4), run="a")
        ref3 = DatasetRef(self.datasetType, dict(instrument="DummyCam", visit=104), run="c")

        # This will sort them on run before visit
        sort = sorted([ref3, ref1, ref2, ref4])
        self.assertEqual(sort, [ref2, ref4, ref1, ref3], msg=f"Got order: {[r.dataId for r in sort]}")

        # Now with strings
        with self.assertRaises(TypeError):
            sort = sorted(["z", ref1, "c"])

    def testOverrideStorageClass(self):
        storageA = StorageClass("test_a", pytype=list)

        ref = DatasetRef(self.datasetType, self.dataId, run="somerun")

        ref_new = ref.overrideStorageClass(storageA)
        self.assertNotEqual(ref, ref_new)
        self.assertEqual(ref_new.datasetType.storageClass, storageA)
        self.assertEqual(ref_new.overrideStorageClass(ref.datasetType.storageClass), ref)
        self.assertTrue(ref.is_compatible_with(ref_new))
        self.assertFalse(ref_new.is_compatible_with(None))

        # Check different code paths of incompatibility.
        ref_incompat = DatasetRef(ref.datasetType, ref.dataId, run="somerun2", id=ref.id)
        self.assertFalse(ref.is_compatible_with(ref_incompat))  # bad run
        ref_incompat = DatasetRef(ref.datasetType, ref.dataId, run="somerun")
        self.assertFalse(ref.is_compatible_with(ref_incompat))  # bad ID

        incompatible_sc = StorageClass("my_int", pytype=int)
        with self.assertRaises(ValueError):
            # Do not test against "ref" because it has a default storage class
            # of "object" which is compatible with everything.
            ref_new.overrideStorageClass(incompatible_sc)

    def testPickle(self):
        ref = DatasetRef(self.datasetType, self.dataId, run="somerun")
        s = pickle.dumps(ref)
        self.assertEqual(pickle.loads(s), ref)

    def testJson(self):
        ref = DatasetRef(self.datasetType, self.dataId, run="somerun")
        s = ref.to_json()
        self.assertEqual(DatasetRef.from_json(s, universe=self.universe), ref)

    def testFileDataset(self):
        ref = DatasetRef(self.datasetType, self.dataId, run="somerun")
        file_dataset = FileDataset(path="something.yaml", refs=ref)
        self.assertEqual(file_dataset.refs, [ref])

        ref2 = DatasetRef(self.datasetType, self.dataId, run="somerun2")
        with self.assertRaises(ValueError):
            FileDataset(path="other.yaml", refs=[ref, ref2])


if __name__ == "__main__":
    unittest.main()
