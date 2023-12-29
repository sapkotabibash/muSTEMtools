#This code is copied from abtem package. https://github.com/abTEM/abTEM
from ase import Atoms
from ase.build.tools import rotation_matrix, cut
from ase.cell import Cell
import numpy as np


def is_cell_orthogonal(cell: Atoms | Cell | np.ndarray, tol: float = 1e-12):
    """
    Check whether atoms have an orthogonal cell. 


    Parameters
    ----------
    cell : ase.Atoms
        The atoms that should be checked.
    tol : float
        Components of the lattice vectors below this value are considered to be zero.

    Returns
    -------
    orthogonal : bool
        True if cell is orthogonal.
    """
    if hasattr(cell, "cell"):
        cell = cell.cell

    return not np.any(np.abs(cell[~np.eye(3, dtype=bool)]) > tol)





def orthogonalize_cell(
    atoms: Atoms,
    max_repetitions: int = 5,
    return_transform: bool = False,
    allow_transform: bool = True,
    plane: str | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    box: tuple[float, float, float] = None,
    tolerance: float = 0.01,
):
    """
    Make the cell of the given atoms orthogonal. This is accomplished by repeating the cell until lattice vectors
    are close to the three principal Cartesian directions. If the structure is not exactly orthogonal after the
    structure is repeated by a given maximum number, the remaining difference is made up by applying strain.
   
    Parameters
    ----------
    atoms : ase.Atoms
        The non-orthogonal atoms.
    max_repetitions : int
        The maximum number of repetitions allowed. Increase this to allow more repetitions and hence less strain.
    return_transform : bool
        If true, return the transformations that were applied to make the atoms orthogonal.
    allow_transform : bool
        If false no transformation is applied to make the cell orthogonal, hence a non-orthogonal cell may be returned.
        plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If given as a string, it must be a concatenation of two of `x`, `y`
        and `z`; the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If given as a string, it must be a concatenation of two of `x`, `y`
        and `z`; the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential. This is equivalent to
        translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined from the atoms' cell.
        If the box size does not match an integer number of the atoms' supercell, an affine transformation may be
        necessary to preserve periodicity, determined by the `periodic` keyword.
    tolerance : float
        Determines what is defined as a plane. All atoms within a distance equal to tolerance [Å] from a given plane
        will be considered to belong to that plane.

    Returns
    -------
    atoms : ase.Atoms
        The orthogonal atoms.
    transform : tuple of arrays, optional
        The applied transform given as Euler angles (by default not returned).
    """

    cell = atoms.cell
    cell[np.abs(cell) < 1e-6] = 0.0
    atoms.set_cell(cell)
    atoms.wrap()

    if origin != (0.0, 0.0, 0.0):
        atoms.translate(-np.array(origin))
        atoms.wrap()

    if plane != "xy":
        atoms = rotate_atoms_to_plane(atoms, plane)

    if box is None:
        box = best_orthogonal_cell(atoms.cell, max_repetitions=max_repetitions)

    if tuple(np.diag(atoms.cell)) == tuple(box):
        return atoms

    if np.any(atoms.cell.lengths() < tolerance):
        raise RuntimeError("Cell vectors must have non-zero length.")

    inv = np.linalg.inv(atoms.cell)
    vectors = np.dot(np.diag(box), inv)
    vectors = np.round(vectors)

    atoms = cut(atoms, *vectors, tolerance=tolerance)

    A = np.linalg.solve(atoms.cell.complete(), np.diag(box))

    if allow_transform:
        atoms.positions[:] = np.dot(atoms.positions, A)
        atoms.cell[:] = np.diag(box)

    elif not np.allclose(A, np.eye(3)):
        raise RuntimeError()

    if return_transform:
        rotation, scale, shear = decompose_affine_transform(A)
        return atoms, (np.array(rotation_matrix_to_euler(rotation)), scale, shear)
    else:
        return atoms

def best_orthogonal_cell(
    cell: np.ndarray, max_repetitions: int = 5, eps: float = 1e-12
) -> np.ndarray:
    """
    Find the closest orthogonal cell for a given cell given a maximum number of repetitions in all directions.

    Parameters
    ----------
    cell : np.ndarray
        Cell of dimensions 3x3.
    max_repetitions : int
        Maximum number of allowed repetitions (default is 5).
    eps : float
        Lattice vector components below this value are considered to be zero.

    Returns
    -------
    cell : np.ndarray
        Closest orthogonal cell found.
    """
    zero_vectors = np.linalg.norm(cell, axis=0) < eps

    if zero_vectors.sum() > 1:
        raise RuntimeError(
            "Two or more lattice vectors of the provided `Atoms` object have no length."
        )

    if isinstance(max_repetitions, int):
        max_repetitions = (max_repetitions,) * 3

    nx = np.arange(-max_repetitions[0], max_repetitions[0] + 1)
    ny = np.arange(-max_repetitions[1], max_repetitions[1] + 1)
    nz = np.arange(-max_repetitions[2], max_repetitions[2] + 1)

    a, b, c = cell
    vectors = np.abs(
        (
            (nx[:, None] * a[None])[:, None, None]
            + (ny[:, None] * b[None])[None, :, None]
            + (nz[:, None] * c[None])[None, None, :]
        )
    )

    norm = np.linalg.norm(vectors, axis=-1)
    nonzero = norm > eps
    norm[nonzero == 0] = eps

    new_vectors = []
    for i in range(3):
        angles = vectors[..., i] / norm

        small_angles = np.abs(angles.max() - angles < eps)

        small_angles = np.where(small_angles * nonzero)

        shortest_small_angles = np.argmin(np.linalg.norm(vectors[small_angles], axis=1))

        new_vector = np.array(
            [
                nx[small_angles[0][shortest_small_angles]],
                ny[small_angles[1][shortest_small_angles]],
                nz[small_angles[2][shortest_small_angles]],
            ]
        )

        new_vector = np.sign(np.dot(new_vector, cell)[i]) * new_vector
        new_vectors.append(new_vector)

    cell = np.dot(new_vectors, np.array(cell))
    return np.linalg.norm(cell, axis=0)


def orthogonalize_cell(
    atoms: Atoms,
    max_repetitions: int = 5,
    return_transform: bool = False,
    allow_transform: bool = True,
    plane: str | tuple[tuple[float, float, float], tuple[float, float, float]] = "xy",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    box: tuple[float, float, float] = None,
    tolerance: float = 0.01,
):
    """
    Make the cell of the given atoms orthogonal. This is accomplished by repeating the cell until lattice vectors
    are close to the three principal Cartesian directions. If the structure is not exactly orthogonal after the
    structure is repeated by a given maximum number, the remaining difference is made up by applying strain.

    Parameters
    ----------
    atoms : ase.Atoms
        The non-orthogonal atoms.
    max_repetitions : int
        The maximum number of repetitions allowed. Increase this to allow more repetitions and hence less strain.
    return_transform : bool
        If true, return the transformations that were applied to make the atoms orthogonal.
    allow_transform : bool
        If false no transformation is applied to make the cell orthogonal, hence a non-orthogonal cell may be returned.
        plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If given as a string, it must be a concatenation of two of `x`, `y`
        and `z`; the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If given as a string, it must be a concatenation of two of `x`, `y`
        and `z`; the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential. This is equivalent to
        translating the atoms. The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined from the atoms' cell.
        If the box size does not match an integer number of the atoms' supercell, an affine transformation may be
        necessary to preserve periodicity, determined by the `periodic` keyword.
    tolerance : float
        Determines what is defined as a plane. All atoms within a distance equal to tolerance [Å] from a given plane
        will be considered to belong to that plane.

    Returns
    -------
    atoms : ase.Atoms
        The orthogonal atoms.
    transform : tuple of arrays, optional
        The applied transform given as Euler angles (by default not returned).
    """

    cell = atoms.cell
    cell[np.abs(cell) < 1e-6] = 0.0
    atoms.set_cell(cell)
    atoms.wrap()

    if origin != (0.0, 0.0, 0.0):
        atoms.translate(-np.array(origin))
        atoms.wrap()

    if plane != "xy":
        atoms = rotate_atoms_to_plane(atoms, plane)

    if box is None:
        box = best_orthogonal_cell(atoms.cell, max_repetitions=max_repetitions)

    if tuple(np.diag(atoms.cell)) == tuple(box):
        return atoms

    if np.any(atoms.cell.lengths() < tolerance):
        raise RuntimeError("Cell vectors must have non-zero length.")

    inv = np.linalg.inv(atoms.cell)
    vectors = np.dot(np.diag(box), inv)
    vectors = np.round(vectors)

    atoms = cut(atoms, *vectors, tolerance=tolerance)

    A = np.linalg.solve(atoms.cell.complete(), np.diag(box))

    if allow_transform:
        atoms.positions[:] = np.dot(atoms.positions, A)
        atoms.cell[:] = np.diag(box)

    elif not np.allclose(A, np.eye(3)):
        raise RuntimeError()

    if return_transform:
        rotation, scale, shear = decompose_affine_transform(A)
        return atoms, (np.array(rotation_matrix_to_euler(rotation)), scale, shear)
    else:
        return atoms





