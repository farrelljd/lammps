/* ----------------------------------------------------------------------
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

PairStyle(superparamagnetic/sf,PairSuperparamagneticSF)

#else

#ifndef LMP_PAIR_SUPERPARAMAGNETIC_SF_H
#define LMP_PAIR_SUPERPARAMAGNETIC_SF_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSuperparamagneticSF : public Pair {
 public:
  PairSuperparamagneticSF(class LAMMPS *);
  virtual ~PairSuperparamagneticSF();
  virtual void compute(int, int);
  void compute_forces(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  //double memory_usage();

 protected:
  int nmax, iterstep, simstep;
  double cut_lj_global,cut_coul_global;
  double **cut_lj,**cut_ljsq;
  double **cut_coul,**cut_coulsq;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4;
  double **scale;
  double scf_energy;
  double  b[3], field[3], oscillating, chi, tolerance;
  int component;
  enum {X_COMPONENT=0, Y_COMPONENT=1, Z_COMPONENT=2};

  void allocate();

 private:
  int update_dipoles();
};

}

#endif
#endif
