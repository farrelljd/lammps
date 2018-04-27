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

#ifdef PAIR_CLASS

PairStyle(lj/cut/induced-dipole/cut/memory,PairLJCutInducedDipoleCutMemory)

#else

#ifndef LMP_PAIR_LJ_CUT_INDUCED_DIPOLE_CUT_MEMORY_H
#define LMP_PAIR_LJ_CUT_INDUCED_DIPOLE_CUT_MEMORY_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutInducedDipoleCutMemory : public Pair {
 public:
  PairLJCutInducedDipoleCutMemory(class LAMMPS *);
  virtual ~PairLJCutInducedDipoleCutMemory();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  //double memory_usage();

 protected:
  double cut_lj_global,cut_coul_global;
  double **cut_lj,**cut_ljsq,**cut_ljsq3inv,**cut_ljsq4inv;
  double **cut_coul,**cut_coulsq;
  double **epsilon,**sigma, **factor;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double ***distances;

  int nmax, iterstep, simstep, component;
  double scf_energy;
  double  b[3], field[3], oscillating, chi, tolerance;
  enum {X_COMPONENT=0, Y_COMPONENT=1, Z_COMPONENT=2};

  void allocate();

 private:
  int update_dipoles();
  void compute_distances();
  void compute_forces(int, int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args in pair_style command

Self-explanatory.

E: Cannot (yet) use 'electron' units with dipoles

This feature is not yet supported.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair dipole/cut requires atom attributes q, mu, torque

The atom style defined does not have these attributes.

*/
