############################################################################
#
# potfit -- The ITAP Force Matching Program
# 	Copyright 2002-2014
#
# 	Institute for Theoretical and Applied Physics,
# 	University of Stuttgart, D-70550 Stuttgart, Germany
# 	http://potfit.sourceforge.net/
#
############################################################################
#
#   This file is part of potfit.
#
#   potfit is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   potfit is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with potfit; if not, see <http://www.gnu.org/licenses/>.
#
############################################################################
#
# Beware: This Makefile works only with GNU make (gmake)!
#
# Usage:  make <target>
#
# <target> has the form
#
#    potfit[_<parallel>][_<option>[_<option>...]]
#
# The parallelization method <parallel> can be:
#
#    mpi   compile for parallel execution, using MPI
#
###########################################################################
#
# Customizing this Makefile
#
# As potfit supports a large number of compile options, you will have to
# compile potfit freqently. Before doing so, however, you must check whether
# the settings in this Makefile fit your needs. You possibly have to
# customize these setttings. Before you can do that, we have to explain
# a bit how the compilation process works.
#
# The compilation process requires the SYSTEM variable in the Makefile to be
# set to any of the predefined values. It specifies what system you have, and
# what compiler you are using. The flags for the compiler and the linker
# are then selected as a function of this variable.
#
# Another important ingredient is the parallelization method, which is
# determined from the make target. The parallelization method is stored
# in the variable PARALLEL, which takes as value SERIAL or MPI.
#
# Depending on the value of ${SYSTEM}, a number of variables must be
# set, from which everything else is constructed.
#
# CC_SERIAL defines the compiler for serial compilation, CC_MPI the one
# to be used for parallelization
#
# BIN_DIR defines the directory where the potfit binary is put. Note that
# this directory must exist.
#
# MV defines the program used to move the potfit binary to ${BIN_DIR}.
# The default is mv, which is usually ok.
#
# The compilation options are stored in the variable CFLAGS.
# The initial value of CFLAGS is set to the variable FLAGS,
# which can be given on the command line.
#
# If the option debug was specified, ${DEBUG_FLAGS} is then appended
# to ${CFLAGS}, otherwise ${OPT_FLAGS}. If the option prof was specified
# (for profiling), ${PROF_FLAGS} is also appended to ${CFLAGS}. However,
# before appending ${OPT_FLAGS} or ${DEBUG_FLAGS} to ${CFLAGS}, some
# parallelization specific flags are appended to them:
#
#   OPT_FLAGS   += ${${PARALLEL}_FLAGS} ${OPT_${PARALLEL}_FLAGS}
#   DEBUG_FLAGS += ${${PARALLEL}_FLAGS} ${DEBUG_${PARALLEL}_FLAGS}
#
# If any of these variables is not defined, it is assumed to be empty.
# This setup should provide sufficient flexibility to set one's favorite
# flags, depending on parallelization, profiling, and optimization/debugging.
#
# Similarly, the link libraries are stored in the variable LIBS,
# to which ${${PARALLEL}_LIBS} and possibly ${PROF_LIBS} (for profiling)
# is appended.
#
# You may have to change the setting for an existing value of SYSTEM.
# or you have to add support for a new value of SYSTEM. The latter is
# best done by using the folloing template for SYSTEM=custom:
#
# ifeq (custom,${SYSTEM})
#   CC_SERIAL		= serial-compiler
#   CC_MPI		= MPI-compiler
#   OMPI_CC      	= compiler for mpicc
#   OMPI_CLINKER 	= linker for mpicc
#   OPT_FLAGS		+= generic flags for optimization
#   DEBUG_FLAGS		+= generic flags for debugging
#   PROF_FLAGS		+= flags for profiling
#   PROF_LIBS		+= libraries for profiling
#   LFLAGS_SERIAL 	+= flags for serial linking
#   LFLAGS_MPI 		+= flags for MPI linking
#   export        MPICH_CC MPICH_CLINKER
# endif
#
# Variables remaining empty need not be mentioned.

###########################################################################
#
#  Set the shell for systems where bash is not the default
#
###########################################################################

SHELL = /bin/bash

###########################################################################
#
#  Adjust these variables to your system
#
###########################################################################

# Currently the following systems are available:
# x86_64-icc  	64bit Intel Compiler
# x86_64-gcc    64bit GNU Compiler
# x86_64-clang 	64bit LLVM Compiler
# i686-icc 	32bit Intel Compiler
# i686-gcc  	32bit GNU Compiler
#
#SYSTEM 		= x86_64-icc 	# Use this as fallback
SYSTEM 		= $(shell uname -m)-gcc

# This is the directory where the potfit binary will be moved to.
# If BIN_DIR is empty, the binary will not be moved.
BIN_DIR 	= bin

# Base directory of your installation of the MKL or ACML

# General settings
MKLDIR          = /opt/intel/composer_xe_2013.3.163/mkl
ACML4DIR  	= /opt/acml4.4.0/gfortran64
ACML5DIR  	= /opt/acml/gfortran64
LIBMDIR 	= /opt/acml/libm

uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
ifeq (${uname_S},Darwin)
  TARGET = MACOS
else
  TARGET = LINUX
endif

# ITAP settings
#BIN_DIR 	= ${HOME}/bin/i386-linux
#MKLDIR          = /common/linux/paket/intel/compiler-11.0/cc/mkl
#ACML4DIR  	= /common/linux/paket/acml4.4.0/gfortran64

###########################################################################
#
#  Defaults for some variables
#
###########################################################################

STRIP 		= $(shell which strip 2> /dev/null)
LIBS		+= -lm
MPI_FLAGS	+= -DMPI
DEBUG_FLAGS	+= -DDEBUG
ACML4PATH 	= ${ACML4DIR}/lib
ACML5PATH 	= ${ACML5DIR}/lib

###########################################################################
#
#  flags for 64bit
#
###########################################################################

ifeq (x86_64-icc,${SYSTEM})
# compiler
  CC_SERIAL     = icc
  CC_MPI        = mpicc
  OMPI_CC       = icc
  OMPI_CLINKER  = icc

# general optimization flags
  OPT_FLAGS     += -fast -xHost

# profiling and debug flags
  PROF_FLAGS    += --profile-functions
  PROF_LIBS     += --profile-functions
  DEBUG_FLAGS   += -g -Wall

# MACOS Math Kernel Library
ifeq (${TARGET},MACOS)
   LIBS		 += -framework Accelerate

# Intel Math Kernel Library
else ifeq (,$(strip $(findstring acml,${MAKETARGET})))
  CINCLUDE 	+= -I${MKLDIR}/include
  LIBS 		+= -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential \
         	   -lmkl_core -Wl,--end-group -lpthread

# AMD Core Math Library
else ifneq (,$(strip $(findstring acml4,${MAKETARGET})))
  CINCLUDE 	+= -I${ACML4DIR}/include
  LIBS		= -L${ACML4PATH} -lpthread -lacml -lacml_mv
else ifneq (,$(strip $(findstring acml5,${MAKETARGET})))
   LIBMPATH 	= ${LIBMDIR}/lib/dynamic
   CINCLUDE     += -I${ACML5DIR}/include -I${LIBMDIR}/include
   LIBS		+= -L${ACML5PATH} -L${LIBMPATH} -lpthread -lacml -lamdlibm

endif

 export        OMPI_CC OMPI_CLINKER
endif

ifeq (x86_64-gcc,${SYSTEM})
# compiler
  CC_SERIAL     = gcc-6
  CC_MPI        = mpicc
  OMPI_CC       = gcc-6
  OMPI_CLINKER  = gcc-6

# general optimization flags
  OPT_FLAGS     += -O3 -march=native -Wno-unused

# profiling and debug flags
  PROF_FLAGS    += -g3 -pg
  PROF_LIBS     += -g3 -pg
  DEBUG_FLAGS   += -g3 -Wall

# MACOS Math Kernel Library
ifeq (${TARGET},MACOS)
  OPT_FLAGS    += -Wa,-q
  LIBS         += -framework Accelerate

# Intel Math Kernel Library
else ifeq (,$(strip $(findstring acml,${MAKETARGET})))
  CINCLUDE      += -I${MKLDIR}/include
  LIBS 		+= -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \

# AMD Core Math Library
else ifneq (,$(strip $(findstring acml4,${MAKETARGET})))
  CINCLUDE     	+= -I${ACML4DIR}/include
  LIBS		+= -L${ACML4PATH} -lpthread -lacml -lacml_mv -Wl,--as-needed
else ifneq (,$(strip $(findstring acml5,${MAKETARGET})))
  LIBMPATH 	= ${LIBMDIR}/lib/dynamic
  CINCLUDE     	+= -I${ACML5DIR}/include -I${LIBMDIR}/include
  LIBS		+= -L${ACML5PATH} -L${LIBMPATH} -lpthread -lacml -lamdlibm -Wl,--as-needed

endif

 export        OMPI_CC OMPI_CLINKER
endif

ifeq (x86_64-clang,${SYSTEM})
# compiler
  CC_SERIAL     = clang
  CC_MPI        = mpicc
  OMPI_CC       = clang
  OMPI_CLINKER  = clang

# general optimization flags
  OPT_FLAGS     += -O3 -march=native -std=gnu99

# profiling and debug flags
  PROF_FLAGS    += -g
  PROF_LIBS     += -g
  DEBUG_FLAGS   += -g -Wall -Werror -pedantic -std=gnu99
  ASAN_FLAGS 	+= -g -fsanitize=address -fno-omit-frame-pointer
  ASAN_LFLAGS 	= -g -fsanitize=address

# MACOS Math Kernel Library
ifeq (${TARGET},MACOS)
   LIBS          += -framework Accelerate

# Intel Math Kernel Library
else ifeq (,$(strip $(findstring acml,${MAKETARGET})))
  CINCLUDE      += -I${MKLDIR}/include
  LIBS 		+= -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \

# AMD Core Math Library
else ifneq (,$(strip $(findstring acml4,${MAKETARGET})))
  CINCLUDE     	+= -I${ACML4DIR}/include
  LIBS		+= -L${ACML4PATH} -lpthread -lacml -lacml_mv -Wl,--as-needed
else ifneq (,$(strip $(findstring acml5,${MAKETARGET})))
  LIBMPATH 	= ${LIBMDIR}/lib/dynamic
  CINCLUDE     	+= -I${ACML5DIR}/include -I${LIBMDIR}/include
  LIBS		+= -L${ACML5PATH} -L${LIBMPATH} -lpthread -lacml -lamdlibm -Wl,--as-needed

endif

 export        OMPI_CC OMPI_CLINKER
endif

###########################################################################
#
#  flags for 32bit
#
###########################################################################

ifeq (i686-icc,${SYSTEM})
# compiler
  CC_SERIAL	= icc
  CC_MPI	= mpicc
  OMPI_CC       = icc
  OMPI_CLINKER  = icc

# general optimization flags
  OPT_FLAGS	+= -fast -xHost

# profiling and debug flags
  PROF_FLAGS	+= -prof-gen
  PROF_LIBS 	+= -prof-gen
  DEBUG_FLAGS	+= -g -Wall -wd981 -wd1572

# MACOS Math Kernel Library
ifeq (${TARGET},MACOS)
   LIBS += -framework Accelerate

# Intel Math Kernel Library
else ifeq (,$(strip $(findstring acml,${MAKETARGET})))
  CINCLUDE      += -I${MKLDIR}/include
  LIBS 		+= -Wl,--start-group -lmkl_intel -lmkl_sequential -lmkl_core \

# AMD Core Math Library
else ifneq (,$(strip $(findstring acml4,${MAKETARGET})))
  CINCLUDE     	+= -I$(ACML4DIR)/include
  LIBS		+= -L${ACML4PATH} -lpthread -lacml
else ifneq (,$(strip $(findstring acml5,${MAKETARGET})))
  LIBMPATH 	= ${LIBMDIR}/lib/dynamic
  CINCLUDE     	+= -I$(ACML5DIR)/include -I${LIBMDIR}/include
  LIBS		+= -L${ACML5PATH} -L${LIBMPATH} -lpthread -lacml

endif

  export        OMPI_CC OMPI_CLINKER
endif

ifeq (i686-gcc,${SYSTEM})
# compiler
  CC_SERIAL	= gcc
  CC_MPI	= mpicc
  OMPI_CC     	= gcc
  OMPI_CLINKER 	= gcc

# general optimization flags
  OPT_FLAGS	+= -O3 -march=native -Wno-unused

# profiling and debug flags
  PROF_FLAGS	+= -g3 -pg
  PROF_LIBS	+= -g3 -pg
  DEBUG_FLAGS	+= -g3 -Wall

# MACOS Math Kernel Library
ifeq (${TARGET},MACOS)
   OPT_FLAGS    += -Wa,-q
   LIBS         += -framework Accelerate

# Intel Math Kernel Library
else ifeq (,$(strip $(findstring acml,${MAKETARGET})))
  CINCLUDE      += -I${MKLDIR}/include
  LIBS		+= -Wl,--start-group -lmkl_intel -lmkl_sequential -lmkl_core \
   		   -Wl,--end-group -lpthread -Wl,--as-needed
# AMD Core Math Library
else ifneq (,$(strip $(findstring acml4,${MAKETARGET})))
  CINCLUDE     	+= -I$(ACML4DIR)/include
  LIBS		+= -L${ACML4PATH} -lpthread -lacml -Wl,--as-needed

else ifneq (,$(strip $(findstring acml5,${MAKETARGET})))
  LIBMPATH 	= ${LIBMDIR}/lib/dynamic
  CINCLUDE     	+= -I$(ACML5DIR)/include -I${LIBMDIR}/include
  LIBS		+= -L${ACML5PATH} -L${LIBMPATH} -lpthread -lacml -Wl,--as-needed

endif

  export        OMPI_CC OMPI_CLINKER
endif


###########################################################################
#
#  Parallelization method
#
###########################################################################

# default is serial
PARALLEL = SERIAL
# MPI
ifneq (,$(strip $(findstring mpi,${MAKETARGET})))
PARALLEL = MPI
endif


###########################################################################
#
#  Compiler, flags, libraries
#
###########################################################################

# compiler; if empty, we issue an error later
CC = ${CC_${PARALLEL}}

# optimization flags
OPT_FLAGS   += ${${PARALLEL}_FLAGS} ${OPT_${PARALLEL}_FLAGS} -DNDEBUG
DEBUG_FLAGS += ${${PARALLEL}_FLAGS} ${DEBUG_${PARALLEL}_FLAGS}

# libraries
LIBS += ${${PARALLEL}_LIBS}

# optimization or debug
CFLAGS := ${FLAGS}
ifneq (,$(findstring debug,${MAKETARGET}))
CFLAGS += ${DEBUG_FLAGS}
else
ifneq (,$(findstring asan,${MAKETARGET}))
CFLAGS += ${ASAN_FLAGS}
LFLAGS_SERIAL = ${ASAN_LFLAGS}
LFLAGS_MPI = ${ASAN_LFLAGS}
else
CFLAGS += ${OPT_FLAGS}
endif
endif

# profiling support
ifneq (,$(findstring prof,${MAKETARGET}))
CFLAGS += ${PROF_FLAGS}
LIBS   += ${PROF_LIBS}
endif


###########################################################################
#
# potfit sources
#
###########################################################################

POTFITHDR   	= bracket.h elements.h optimize.h potfit.h potential.h \
		  random.h splines.h utils.h
POTFITSRC 	= bracket.c brent.c config.c elements.c errors.c forces.c linmin.c \
		  param.c potential_input.c potential_output.c potfit.c \
		  powell_lsq.c random.c simann.c splines.c utils.c

ifneq (,$(strip $(findstring pair,${MAKETARGET})))
  POTFITSRC      += force_pair.c
endif

ifneq (,$(strip $(findstring eam,${MAKETARGET})))
  ifneq (,$(strip $(findstring meam,${MAKETARGET})))
    POTFITSRC      += force_meam.c
  else ifneq (,$(strip $(findstring coulomb,${MAKETARGET})))
    POTFITSRC      += force_eam_elstat.c
  else ifneq (,$(strip $(findstring dipole,${MAKETARGET})))
    POTFITSRC      += force_eam_elstat.c
  else
    POTFITSRC      += force_eam.c
  endif
endif

ifneq (,$(strip $(findstring coulomb,${MAKETARGET})))
  ifeq (,$(strip $(findstring eam,${MAKETARGET})))
    POTFITSRC      += force_elstat.c
  endif
endif

ifneq (,$(strip $(findstring dipole,${MAKETARGET})))
  ifeq (,$(strip $(findstring eam,${MAKETARGET})))
    POTFITSRC      += force_elstat.c
  endif
endif

ifneq (,$(strip $(findstring adp,${MAKETARGET})))
  POTFITSRC      += force_adp.c
endif

ifneq (,$(strip $(findstring stiweb,${MAKETARGET})))
  POTFITSRC      += force_stiweb.c
endif

ifneq (,$(strip $(findstring tersoff,${MAKETARGET})))
  POTFITSRC      += force_tersoff.c
endif

ifneq (,$(strip $(findstring apot,${MAKETARGET})))
  POTFITHDR      += functions.h
  POTFITSRC      += functions.c
  ifneq (,$(strip $(findstring pair,${MAKETARGET})))
    POTFITSRC      += chempot.c
  endif
else
  ifneq (,$(strip $(findstring meam,${MAKETARGET})))
    POTFITSRC 	+= rescale_meam.c
  else
    POTFITSRC      += rescale.c
  endif
endif

ifneq (,$(strip $(findstring evo,${MAKETARGET})))
  POTFITSRC      += diff_evo.c
endif

ifneq (,$(strip $(findstring parab,${MAKETARGET})))
  POTFITSRC      += parabola.c
endif

MPISRC          = mpi_utils.c

#########################################################
#
# potfit Configuration rules
#
#########################################################

HEADERS := ${POTFITHDR}

# serial or mpi
ifneq (,$(strip $(findstring mpi,${MAKETARGET})))
SOURCES	:= ${POTFITSRC} ${MPISRC}
else
SOURCES	:= ${POTFITSRC}
endif

###  INTERACTIONS  #######################################

INTERACTION = 0

# pair potentials
ifneq (,$(findstring pair,${MAKETARGET}))
  CFLAGS += -DPAIR
  INTERACTION = 1
endif

# embedded atom method (EAM) potentials
ifneq (,$(strip $(findstring eam,${MAKETARGET})))
  ifneq (,$(findstring 1,${INTERACTION}))
    ERROR += More than one potential model specified
  endif
  ifneq (,$(strip $(findstring meam,${MAKETARGET})))
    CFLAGS  += -DMEAM
  else ifneq (,$(strip $(findstring tbeam,${MAKETARGET})))
    CFLAGS  += -DTBEAM
  else
    CFLAGS  += -DEAM
  endif
  INTERACTION = 1
endif

# COULOMB
ifneq (,$(strip $(findstring coulomb,${MAKETARGET})))
  ifeq (,$(strip $(findstring eam,${MAKETARGET})))
    ifneq (,$(findstring 1,${INTERACTION}))
      ERROR += More than one potential model specified
    endif
  endif
  ifeq (,$(strip $(findstring apot,${MAKETARGET})))
    ERROR += COULOMB does not support tabulated potentials
  endif
  CFLAGS  += -DCOULOMB
  INTERACTION = 1
endif

# DIPOLE
ifneq (,$(strip $(findstring dipole,${MAKETARGET})))
  ifeq (,$(strip $(findstring eam,${MAKETARGET})))
    ifneq (,$(findstring 1,${INTERACTION}))
      ERROR += More than one potential model specified
    endif
  endif
  ifeq (,$(strip $(findstring apot,${MAKETARGET})))
    ERROR += DIPOLE does not support tabulated potentials
  endif
  CFLAGS  += -DCOULOMB -DDIPOLE
  INTERACTION = 1
endif

# angular dependent potentials (ADP)
ifneq (,$(strip $(findstring adp,${MAKETARGET})))
  ifneq (,$(findstring 1,${INTERACTION}))
    ERROR += More than one potential model specified
  endif
  CFLAGS  += -DADP
  INTERACTION = 1
endif

# Stillinger-Weber potentials (STIWEB)
ifneq (,$(strip $(findstring stiweb,${MAKETARGET})))
  ifneq (,$(findstring 1,${INTERACTION}))
    ERROR += More than one potential model specified
  endif
  ifeq (,$(findstring apot,${MAKETARGET}))
    ERROR += STIWEB does not work without the apot flag
  endif
  CFLAGS  += -DSTIWEB
  INTERACTION = 1
endif

# Tersoff potentials (TERSOFF and TERSOFFMOD)
ifneq (,$(strip $(findstring tersoff,${MAKETARGET})))
  ifneq (,$(findstring 1,${INTERACTION}))
    ERROR += "More than one potential model specified."
  endif
  ifeq (,$(findstring apot,${MAKETARGET}))
    ERROR += "TERSOFF does not work without the apot flag!"
  endif
  ifneq (,$(findstring tersoffmod,${MAKETARGET}))
    CFLAGS += -DTERSOFFMOD
  endif
  CFLAGS  += -DTERSOFF
  INTERACTION = 1
endif

ifneq (,$(findstring 0,${INTERACTION}))
  ERROR += "No interaction model specified."
endif

# EVO - for differential evolution
ifneq (,$(findstring evo,${MAKETARGET}))
CFLAGS += -DEVO
endif

# APOT - for analytic potentials
ifneq (,$(findstring apot,${MAKETARGET}))
  ifneq (,$(findstring resc,${MAKETARGET}))
    ERROR += "Analytic potentials are incompatible with the rescale option!"
  endif
CFLAGS += -DAPOT
endif

# Stress
ifneq (,$(findstring stress,${MAKETARGET}))
CFLAGS += -DSTRESS
endif

# Disable gauge punishments for EAM/ADP
ifneq (,$(findstring nopunish,${MAKETARGET}))
CFLAGS += -DNOPUNISH
endif

ifneq (,$(findstring limit,${MAKETARGET}))
WARNING += "limit is now mandatory -- "
endif

ifneq (,$(findstring parab,${MAKETARGET}))
ERROR += "parab is no longer supported, please remove it from your target -- "
endif

ifneq (,$(findstring wzero,${MAKETARGET}))
ERROR += "wzero is no longer supported, please remove it from your target -- "
endif

ifneq (,$(findstring dist,${MAKETARGET}))
ifeq (,$(findstring MPI,${PARALLEL}))
CFLAGS += -DPDIST
else
ERROR += "dist is not mpi parallelized -- "
endif
endif

ifneq (,$(findstring newscale,${MAKETARGET}))
ERROR += "newscale is no longer supported, please remove it from your target -- "
endif

ifneq (,$(findstring fweight,${MAKETARGET}))
CFLAGS += -DFWEIGHT
endif

ifneq (,$(findstring contrib,${MAKETARGET}))
CFLAGS += -DCONTRIB
endif

# force acml4 or acml5 over acml
ifneq (,$(findstring acml,${MAKETARGET}))
ifeq (,$(findstring acml4,${MAKETARGET}))
ifeq (,$(findstring acml5,${MAKETARGET}))
ERROR += The acml target is obsolete. Please use acml4 or acml5.
endif
endif
endif

ifneq (,$(findstring acml4,${MAKETARGET}))
CFLAGS += -DACML -DACML4
endif

ifeq (${TARGET},MACOS)
CFLAGS += -D__ACCELERATE__
endif

ifneq (,$(findstring acml5,${MAKETARGET}))
CFLAGS += -DACML -DACML5
endif

ifneq (,$(findstring resc,${MAKETARGET}))
CFLAGS += -DRESCALE
endif

# Substitute .o for .c to get the names of the object files
OBJECTS := $(subst .c,.o,${SOURCES})

###########################################################################
#
#	 Rules
#
###########################################################################

# all objects depend on headers
${OBJECTS}: ${HEADERS}

potfit:
	@echo -e "\nError:\tYou cannot compile potfit without any options."
	@echo -e "\tAt least an interaction is required.\n"

# How to compile *.c files
# special rules for force computation
powell_lsq.o: powell_lsq.c
	@echo " [CC] powell_lsq.c"
	@${CC} ${CFLAGS} ${CINCLUDE} -c $< || { \
		echo -e "The following command failed with the above error:\n"; \
		echo -e ${CC} ${CFLAGS} ${CINCLUDE} -c $<"\n"; \
		exit 1; \
		}

# special rules for function evaluation
utils.o: utils.c
	@echo " [CC] utils.c"
	@${CC} ${CFLAGS} ${CINCLUDE} -c $< || { \
		echo -e "The following command failed with the above error:\n"; \
		echo -e ${CC} ${CFLAGS} ${CINCLUDE} -c $<"\n"; \
		exit 1; \
		}

# generic compilation rule
%.o: %.c
ifeq (,${MAKETARGET})
	@echo -e "Usage:"
	@echo -e "  make potfit_[interaction]_[options]\n"
	@echo "For more details on compiling potfit please look at the Makefile"
	@exit
else
	@echo " [CC] ${@:.o=.c}"
	@${CC} ${CFLAGS} -c $< || { \
		echo -e "The following command failed with the above error:\n"; \
		echo -e ${CC} ${CFLAGS} -c $<"\n"; \
		exit 1; \
		}
endif

# How to link
${MAKETARGET}: ${OBJECTS}
	@echo " [LD] $@"
ifeq (,${BIN_DIR})
	@${CC} ${LFLAGS_${PARALLEL}} -o $@ ${OBJECTS} ${LIBS}
else
	@${CC} ${LFLAGS_${PARALLEL}} -o ${BIN_DIR}/$@ ${OBJECTS} ${LIBS}
endif
ifneq (${TARGET},MACOS)
 ifneq (,${STRIP})
   ifeq (,$(findstring prof,${MAKETARGET}))
     ifeq (,$(findstring debug,${MAKETARGET}))
       ifeq (,${BIN_DIR})
 	@${STRIP} --strip-unneeded -R .comment $@
       else
 	@${STRIP} --strip-unneeded -R .comment ${BIN_DIR}/$@
       endif
     endif
   endif
 endif
endif
	@echo -e "Building $@ was sucessfull."

# First recursion only set the MAKETARGET Variable
.DEFAULT:
ifneq (,${CC})
	@${MAKE} --no-print-directory MAKETARGET='$@' STAGE2
else
	@echo "There is no compiler defined for this option."
	@echo -e "Please adjust the Makefile.\n"
	@exit
endif


# Second recursion sets MAKETARGET variable and compiles
# An empty MAKETARGET variable would create an infinite recursion, so we check
STAGE2:
ifneq (,${ERROR})
	@echo -e "\nError: ${ERROR}\n"
else
ifneq (,${MAKETARGET})
ifneq (,${WARNING})
	@echo "${WARNING}"
endif
	@${MAKE} MAKETARGET='${MAKETARGET}' ${MAKETARGET}
else
	@echo 'No TARGET specified.'
endif
endif

###########################################################################
#
#	 Misc. TARGETs
#
###########################################################################

clean:
	rm -f *.o *.u *~ \#* *.V *.T *.O *.il

help:
	@echo "Usage: make potfit[_<parallel>][_<option>[_<option>...]]"

