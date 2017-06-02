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

PairStyle(soft_blob,PairSoftBlob)

#else

#ifndef LMP_PAIR_SOFT_BLOB_H
#define LMP_PAIR_SOFT_BLOB_H

#include "pair.h"
#include <math.h>

namespace LAMMPS_NS {

class PairSoftBlob : public Pair {
 public:
  PairSoftBlob(class LAMMPS *);
  virtual ~PairSoftBlob();

  virtual void compute(int, int);
  void compute_particles(int, int);
  void compute_walls(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  class NeighList *listwall;

 protected:
  double cut_global;
  char id_temp_global[80];
  int walls[2], lower_wall_index, upper_wall_index;
  const int numwalls = 2;
  double *blob_temperature;
  double **cut;
  double **hbb,**wbb,**r0;
  double **r02,**r06,**r012;
  int **pair_type;
  double **soft_blob1;
  double **offset;
  const double wall_colloid_fac = 9.0*cbrt(2)/20.0;

  enum {BLOB_NONE=-1, BLOB_BLOB=1, BLOB_COLLOID=2, BLOB_WALL=3, WALL_WALL=4, WALL_COLLOID=5};

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
