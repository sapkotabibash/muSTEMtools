from ase.build import surface
from ase.io import read, write, mustem
import numpy as np
from atoms_from_abtem import is_cell_orthogonal, orthogonalize_cell


def make_mustem_xtl(lattice, DW = None,keV = 200, indices = (0,0,1), layers = 1, vacuum=None,max_repetitions=10):
    """
    Converts input crystal structure to .xtl file for muSTEM simulation. Also, returns corresponding cif file for the slab.
    
    Parameters
    ------------
    filename: str
        filename for input crystal structure( e.g. CaO.cif). Unit cell must be conventional (not primitive).
    DW: dict
        dictionary with atomic symbol as key and Debye-Waller Factor (B) as value. (eg: {'Ca':0.029,'O':0.012}).
    keV: int
        electron energy in keV.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.
    max_repetitions: int
         The maximum number of repetitions allowed. Increase this to allow more repetitions and hence less strain   
        
    Returns
    ---------------
    NoneType
        
    """
    
    cif = read(lattice) 
    
    if DW == None:
        DW = {key : 0.0 for key in set(cif.get_chemical_symbols()) }
    else:
        slab = surface(cif,indices,layers,vacuum=vacuum,periodic=True)
        
        if not is_cell_orthogonal(slab):
            slab =orthogonalize_cell(slab, max_repetitions=max_repetitions,return_transform=False)
        
        if is_cell_orthogonal(slab):
            outputname = lattice[:-4]+'_' +''.join([*map(str,indices)])
            mustem.write_mustem(file = outputname+'.xtl',atoms = slab, keV = keV,debye_waller_factors=DW)
            write(outputname+'.cif',images = slab)
           
        else:
            raise ValueError("Could not orthogonalize cell.")
    pass
        



