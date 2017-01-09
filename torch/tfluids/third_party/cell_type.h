// The methods in this file borrow heavily from - or or a direct port of - parts
// of the Mantaflow library. Since the Mantaflow library is GNU GPL V3, we are
// releasing this code under GNU as well as a third_party add on. See
// FluidNet/torch/tfluids/third_party/README for more information.

/******************************************************************************
 *
 * MantaFlow fluid solver framework 
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 ******************************************************************************/

#pragma once

// These are the same enum values used in Manta. We can't include grid.h
// from Manta without pulling in the entire library, so we'll just redefine
// them here.
enum CellType {
    TypeNone = 0,
    TypeFluid = 1,
    TypeObstacle = 2,
    TypeEmpty = 4,
    TypeInflow = 8,
    TypeOutflow = 16,
    TypeOpen = 32,
    TypeStick = 128,
    TypeReserved = 256,
    TypeZeroPressure = (1<<15)
};

