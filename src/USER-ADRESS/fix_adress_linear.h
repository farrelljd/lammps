/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(adress/linear,FixAdressLinear)

#else

#ifndef LMP_FIX_ADRESS_LINEAR_H
#define LMP_FIX_ADRESS_LINEAR_H

#include "fix_adress.h"

namespace LAMMPS_NS {

class FixAdressLinear : public FixAdress {
 public:
  double factor;

  FixAdressLinear(class LAMMPS *, int, char **);
  void adress_weight();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E:  defined twice in fix adress command

Self-explanatory.

E: Cannot use fix adress in periodic dimension

Self-explanatory.

E: Cannot use fix adress zlo/zhi for a 2d simulation

Self-explanatory.

E: Variable name for fix adress does not exist

Self-explanatory.

E: Variable for fix adress is invalid style

Only equal-style variables can be used.

W: Should not allow rigid bodies to bounce off relecting walls

LAMMPS allows this, but their dynamics are not computed correctly.

*/
