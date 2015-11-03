/****************************************************************
 *
 * force_elstat.c: Routine used for calculating pair/monopole/dipole
 *     forces/energies in various interpolation schemes.
 *
 ****************************************************************
 *
 * Copyright 2002-2014
 *	Institute for Theoretical and Applied Physics
 *	University of Stuttgart, D-70550 Stuttgart, Germany
 *	http://potfit.sourceforge.net/
 *
 ****************************************************************
 *
 *   This file is part of potfit.
 *
 *   potfit is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   potfit is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with potfit; if not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#include "potfit.h"

#if defined COULOMB && defined CSH

#include "functions.h"
#include "potential.h"
#include "splines.h"
#include "utils.h"

/****************************************************************
 *
 *  compute forces using pair potentials with spline interpolation
 *
 *  returns sum of squares of differences between calculated and reference
 *     values
 *
 *  arguments: *xi - pointer to short-range potential
 *             *forces - pointer to forces calculated from potential
 *             flag - used for special tasks
 *
 * When using the mpi-parallelized version of potfit, all processes but the
 * root process jump into this function immediately after initialization and
 * stay in here for an infinite loop, to exit only when a certain flag value
 * is passed from process 0. When a set of forces needs to be calculated,
 * the root process enters the function with a flag value of 0, broadcasts
 * the current potential table xi and the flag value to the other processes,
 * thus initiating a force calculation. Whereas the root process returns with
 * the result, the other processes stay in the loop. If the root process is
 * called with flag value 1, all processes exit the function without
 * calculating the forces.
 * If anything changes about the potential beyond the values of the parameters,
 * e.g. the location of the sampling points, these changes have to be broadcast
 * from rank 0 process to the higher ranked processes. This is done when the
 * root process is called with flag value 2. Then a potsync function call is
 * initiated by all processes to get the new potential from root.
 *
 * xi_opt is the array storing the potential parameters (usually it is the
 *     opt_pot.table - part of the struct opt_pot, but it can also be
 *     modified from the current potential.
 *
 * forces is the array storing the deviations from the reference data, not
 *     only for forces, but also for energies, stresses or dummy constraints
 *     (if applicable).
 *
 * flag is an integer controlling the behaviour of calc_forces_pair.
 *    flag == 1 will cause all processes to exit calc_forces_pair after
 *             calculation of forces.
 *    flag == 2 will cause all processes to perform a potsync (i.e. broadcast
 *             any changed potential parameters from process 0 to the others)
 *             before calculation of forces
 *    all other values will cause a set of forces to be calculated. The root
 *             process will return with the sum of squares of the forces,
 *             while all other processes remain in the function, waiting for
 *             the next communication initiating another force calculation
 *             loop
 *
 ****************************************************************/

double calc_forces(double *xi_opt, double *forces, int flag)
{
  double tmpsum, sum = 0.0;
  int   first, col, ne, size, i = flag;
  double *xi = NULL;
  apot_table_t *apt = &apot_table;
  double charge[ntypes];
  double sum_charges;
  double dp_kappa;

  double ccos;

  angle_t *angle;

  //printf(" \n \n potlen %d ncol %d step %f\n \n ", calc_pot.len,  calc_pot.ncols, calc_pot.step );

  switch (format) {
      case 0:
	xi = calc_pot.table;
	break;
      case 3:			/* fall through */
      case 4:
	xi = xi_opt;		/* calc-table is opt-table */
	break;
      case 5:
	xi = calc_pot.table;	/* we need to update the calc-table */
  }

  ne = apot_table.total_ne_par;
  size = apt->number;

  /* This is the start of an infinite loop */
  while (1) {
    tmpsum = 0.0;		/* sum of squares of local process */

#if defined APOT && !defined MPI
    if (format == 0) {
      apot_check_params(xi_opt);
      update_calc_table(xi_opt, xi, 0);
    }
#endif /* APOT && !MPI */

#ifdef MPI
    /* exchange potential and flag value */
#ifndef APOT
    MPI_Bcast(xi, calc_pot.len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif /* APOT */
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (flag == 1)
      break;			/* Exception: flag 1 means clean up */

#ifdef APOT
    if (myid == 0)
      apot_check_params(xi_opt);
    MPI_Bcast(xi_opt, ndimtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (format == 0)
      update_calc_table(xi_opt, xi, 0);
#else /* APOT */
    /* if flag==2 then the potential parameters have changed -> sync */
    if (flag == 2)
      potsync();
#endif /* APOT */
#endif /* MPI */

    /* local arrays for electrostatic parameters */
    sum_charges = 0;
    for (i = 0; i < ntypes - 1; i++) {
      if (xi_opt[2 * size + ne + i]) {
	charge[i] = xi_opt[2 * size + ne + i];
	sum_charges += apt->ratio[i] * charge[i];
      } else {
	charge[i] = 0.0;
      }
    }
    apt->last_charge = -sum_charges / apt->ratio[ntypes - 1];
    charge[ntypes - 1] = apt->last_charge;
    if (xi_opt[2 * size + ne + ntypes - 1]) {
      dp_kappa = xi_opt[2 * size + ne + ntypes - 1];
    } else {
      dp_kappa = 0.0;
    }

    /* init second derivatives for splines */
    for (col = 0; col < paircol; col++) {
      first = calc_pot.first[col];
      if (format == 3 || format == 0) {
	spline_ed(calc_pot.step[col], xi + first,
	  calc_pot.last[col] - first + 1, *(xi + first - 2), 0.0, calc_pot.d2tab + first);
      } else {			/* format >= 4 ! */
	spline_ne(calc_pot.xcoord + first, xi + first,
	  calc_pot.last[col] - first + 1, *(xi + first - 2), 0.0, calc_pot.d2tab + first);
      }
    }

#ifndef MPI
    myconf = nconf;
#endif /* MPI */

    /* region containing loop over configurations,
       also OMP-parallelized region */
    {
      int   self;
      vector tmp_force;
      int   h, j, k, type1, type2, uf;
#ifdef STRESS
      int   us, stresses;
#endif /* STRESS */
      int   n_i, n_j;
      double fnval, grad, fnval_tail, grad_tail, grad_i, grad_j;
      atom_t *atom;
      neigh_t *neigh, *neigh_j, *neigh_k;
      double angener;
      int   ijk;

      //printf( " col " ); 
      //for (i = 0; i < paircol ; i++) {
      //        printf( " %d " , (int) (apot_table.cweight[i]) ); 
      //}
      //        printf( " \n");

      /* loop over configurations: M A I N LOOP CONTAINING ALL ATOM-LOOPS */
      for (h = firstconf; h < firstconf + myconf; h++) {
	uf = conf_uf[h - firstconf];
#ifdef STRESS
	us = conf_us[h - firstconf];
#endif /* STRESS */
	/* reset energies and stresses */
	forces[energy_p + h] = 0.0;
#ifdef STRESS
	stresses = stress_p + 6 * h;
	for (i = 0; i < 6; i++)
	  forces[stresses + i] = 0.0;
#endif /* STRESS */

	/* F I R S T LOOP OVER ATOMS: reset forces, dipoles */
	for (i = 0; i < inconf[h]; i++) {	/* atoms */
	  n_i = 3 * (cnfstart[h] + i);
	  if (uf) {
	    forces[n_i + 0] = -force_0[n_i + 0];
	    forces[n_i + 1] = -force_0[n_i + 1];
	    forces[n_i + 2] = -force_0[n_i + 2];
	  } else {
	    forces[n_i + 0] = 0.0;
	    forces[n_i + 1] = 0.0;
	    forces[n_i + 2] = 0.0;
	  }
	}			/* end F I R S T LOOP */


	/* SECOND loop: calculate short-range forces if using a core-shell model */

        for (i = 0; i < inconf[h]; i++) {	/* atoms */
	  atom = conf_atoms + i + cnfstart[h] - firstatom;
	  type1 = atom->type;
	  n_i = 3 * (cnfstart[h] + i);
	  for (j = 0; j < atom->num_neigh; j++) {	/* short-range neighbors */
	    neigh = atom->neigh + j;
	    type2 = neigh->type;
	    col = neigh->col[0];

	    /* In small cells, an atom might interact with itself */
	    self = (neigh->nr == i + cnfstart[h]) ? 1 : 0;
	    /* calculate short-range forces if not using core-shell model*/
	    if (neigh->r < calc_pot.end[col]) {

	      if (uf) {
		fnval =
		  splint_comb_dir(&calc_pot, xi, neigh->slot[0], neigh->shift[0], neigh->step[0], &grad);
	      } else {
		fnval = splint_dir(&calc_pot, xi, neigh->slot[0], neigh->shift[0], neigh->step[0]);
	      }
	      /* avoid double counting if atom is interacting with a
	         copy of itself */
	      if (self) {
		fnval *= 0.5;
		grad *= 0.5;
	      }
              //printf("ener_sum: %f  val: %f \n", forces[energy_p + h], fnval );
	      forces[energy_p + h] += fnval;

	      if (uf) {
		tmp_force.x = neigh->dist_r.x * grad;
		tmp_force.y = neigh->dist_r.y * grad;
		tmp_force.z = neigh->dist_r.z * grad;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = reactio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;

#ifdef STRESS
		/* calculate pair stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }
	    }
          }   /* j loop */
        }  /* i loop */

        /* T H I R D loop: calculate short-range and monopole forces,
	   calculate static field- and dipole-contributions */
        for (i = 0; i < inconf[h]; i++) {	/* atoms */
	  atom = conf_atoms + i + cnfstart[h] - firstatom;
	  type1 = atom->type;
	  n_i = 3 * (cnfstart[h] + i);

	  for (j = 0; j < atom->num_couln; j++) {	/* neighbors */
	    neigh = atom->coulneigh + j;
	    type2 = neigh->type;
	    col = neigh->col[0];

            /* updating tail-functions - only necessary with variing kappa */
	    if (!apt->sw_kappa)
	      elstat_lammps_wolf(neigh->r, dp_kappa, &neigh->fnval_el, &neigh->grad_el );

	    /* In small cells, an atom might interact with itself */
	    self = (neigh->nr == i + cnfstart[h]) ? 1 : 0;

	    /* calculate monopole forces */
	    if (neigh->r < dp_cut && (charge[type1] || charge[type2])) {

	      fnval_tail = neigh->fnval_el;
	      grad_tail = neigh->grad_el;

	      grad_i = charge[type2] * grad_tail;
	      if (type1 == type2) {
		grad_j = grad_i;
	      } else {
		grad_j = charge[type1] * grad_tail;
	      }
	      fnval = charge[type1] * charge[type2] * fnval_tail;
	      grad = charge[type1] * grad_i;

              /* check if pair is a core-shell one */
              // if ( type1==0 && type2==4 || type1==2 && type2==5 ) 
              if ( (int) (apot_table.cweight[col]) == 0 ) {
        //         printf("types null %d  %d \n", type1, type2 ); 
                 if (neigh->r <= rcut[type1 * ntypes + type2]) {   /* suppress coulomb contribution from the pair */
                     fnval -= dp_eps * charge[type1] * charge[type2] / atoms[i].coulneigh[j].r;
                     grad=0;
                 }
              }
              //printf("%d   %d  r: %f   coul: %f \n", i, atoms[i].coulneigh[j].nr, atoms[i].coulneigh[j].r, fnval);

	      if (self) {
		grad_i *= 0.5;
		grad_j *= 0.5;
		fnval *= 0.5;
		grad *= 0.5;
	      }

	      forces[energy_p + h] += fnval ;

              //printf("ener_sum: %f  val: %f \n\n", forces[energy_p + h], fnval );

	      if (uf) {
		tmp_force.x = neigh->dist.x * grad;
		tmp_force.y = neigh->dist.y * grad;
		tmp_force.z = neigh->dist.z * grad;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = reactio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;
#ifdef STRESS
		/* calculate coulomb stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }
	    }

	  }			/* loop over neighbours */
        }			/* end T H I R D loop over atoms */


        /* F O U R T H  loop: self energy contributions and sum-up force contributions */
        double qq;
        double e_shift;
        e_shift=erfc(dp_kappa*dp_cut)/dp_cut;
        for (i = 0; i < inconf[h]; i++) {	/* atoms */
     	  atom = conf_atoms + i + cnfstart[h] - firstatom;
     	  type1 = atom->type;
     	  n_i = 3 * (cnfstart[h] + i);
     
     	  /* self energy contributions */
     	  if (charge[type1]) {
     	    qq = charge[type1] * charge[type1];
     	    fnval = dp_eps * qq * (dp_kappa / sqrt(M_PI) + 0.5*e_shift );
     	    forces[energy_p + h] -= fnval;
             //    printf("self ener: %d  %f  shif: %f kpp: %f  pi: %f \n", i, fnval, e_shift, dp_kappa, M_PI);
     
          }
         }
	 

	/* F I F T H  LOOP: Calculate angular forces and energies */
        for (i = 0; i < inconf[h]; i++) {	/* atoms */
	  /* Set pointer to temp atom pointer */
	  atom = conf_atoms + i + cnfstart[h] - firstatom;
	  type1 = atom->type;
	  /* Skip every 3 spots for force array */
	  n_i = 3 * (cnfstart[h] + i);
	  //col = paircol + type1;

	  /* Find the correct column in the potential table for angle part: g_ijk
	     col2 = paircol + + typ1; */

	  /* Loop over every angle formed by neighbors
	     N(N-1)/2 possible combinations
	     Used in computing angular part g_ijk */

	  /* set angl pointer to angl_part of current atom */
	  angle = atom->angle_part;

	  /* Loop over angles */
	  ijk = 0;
	  for (j = 0; j < atom->num_angn - 1; j++) {

	    /* Get pointer to neighbor j */
	    neigh_j = atom->angneigh + j;

	    for (k = j + 1; k < atom->num_angn; k++) {

	      /* Get pointer to neighbor kk */
	      neigh_k = atom->angneigh + k;

	      angle->g = splint_comb_dir(&calc_pot, xi, angle->slot, angle->shift, angle->step, &angle->dg);

	      
//              printf(" @#@  %f slo shif ste %d  %f  %f  \n", calc_pot.begin[ paircol + type1], angle->slot, angle->shift, angle->step );
//              printf(" @#@  %d  %d  %d    %f  %f  %f  \n", neigh_j->nr+1, i+1 , neigh_k->nr+1, angle->theta*180/M_PI, angle->theta , angle->g );

//	      angener += angle->g;
	      forces[energy_p + h] += angle->g ;


	      /* Increase angl pointer */
	      ijk++;
	      angle++;
            } /* k loop */

	  }  /* j loop */
	}  /* end F I F T H loop over atoms */

//        printf(" @# angener %d  %f  \n", type1, angener );


        /* whole energy contributions flow into tmpsum */
        forces[energy_p + h] /= (double)inconf[h];
        forces[energy_p + h] -= force_0[energy_p + h];
        tmpsum += conf_weight[h] * eweight * dsquare(forces[energy_p + h]);

#ifdef STRESS
      /* whole stress contributions flow into tmpsum */
        if (uf && us) {
  	  for (i = 0; i < 6; i++) {
  	    forces[stresses + i] /= conf_vol[h - firstconf];
  	    forces[stresses + i] -= force_0[stresses + i];
  	    tmpsum += conf_weight[h] * sweight * dsquare(forces[stresses + i]);
  	  }
  	}
#endif /* STRESS */
      }				/* end M A I N loop over configurations */
    }				/* parallel region */

    /* dummy constraints (global) */
#ifdef APOT
    /* add punishment for out of bounds (mostly for powell_lsq) */
    if (myid == 0) {
      tmpsum += apot_punish(xi_opt, forces);
    }
#endif /* APOT */

#ifdef MPI
    /* reduce global sum */
    sum = 0.0;
    MPI_Reduce(&tmpsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* gather forces, energies, stresses */
    if (myid == 0) {		/* root node already has data in place */
      /* forces */
      MPI_Gatherv(MPI_IN_PLACE, myatoms, MPI_VECTOR, forces,
	atom_len, atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_DOUBLE, forces + energy_p,
	conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef STRESS
      /* stresses */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_STENS, forces + stress_p,
	conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
#endif /* STRESS */
    } else {
      /* forces */
      MPI_Gatherv(forces + firstatom * 3, myatoms, MPI_VECTOR,
	forces, atom_len, atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(forces + energy_p + firstconf, myconf, MPI_DOUBLE,
	forces + energy_p, conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef STRESS
      /* stresses */
      MPI_Gatherv(forces + stress_p + 6 * firstconf, myconf, MPI_STENS,
	forces + stress_p, conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
#endif /* STRESS */
    }
#else
    sum = tmpsum;		/* global sum = local sum  */
#endif /* MPI */

    /* root process exits this function now */
    if (myid == 0) {
      fcalls++;			/* Increase function call counter */
      if (isnan(sum)) {
#ifdef DEBUG
	printf("\n--> Force is nan! <--\n\n");
#endif /* DEBUG */
	return 10e10;
      } else
	return sum;
    }
  }

  /* once a non-root process arrives here, all is done. */
  return -1.0;
}

#endif /* COULOMB */
