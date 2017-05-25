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

#ifdef BOND_CLASS

BondStyle(soft_blob,BondSoftBlob)

#else

#ifndef LMP_BOND_SOFT_BLOB_H
#define LMP_BOND_SOFT_BLOB_H

#include <stdio.h>
#include <unordered_map>
#include <vector>
#include "bond.h"

namespace LAMMPS_NS {

class BondSoftBlob : public Bond {
 public:
  BondSoftBlob(class LAMMPS *);
  virtual ~BondSoftBlob();
  virtual void get_displacement(int, int, int, double &, double &, double &);
  void init_style();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  double single(int, double, int, int, double &);
  virtual void get_grafting_points();

 protected:
  char id_temp_global[80];
  double *blob_temperature;
  double *k,*r0,***gp,***gp_local;
  int *tf;
  int gpset;
  int gpx_index{-1};
  int gpy_index{-1};
  int gpz_index{-1};
  double *gpx, *gpy, *gpz;
  std::unordered_map<int, std::unordered_map<int, std::vector<double>>> gp_map;

  enum InteractionTypes {BLOB_BLOB=1,BLOB_WALL=2};
  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
