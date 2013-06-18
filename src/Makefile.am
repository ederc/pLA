# Copyright 2012 Christian Eder
# 
# This file is part of F4RT, licensed under the GNU General Public
# License version 3. See COPYING for more information.

AUTOMAKE_OPTIONS = subdir-objects

INCLUDES=-I$(top_srcdir) -I$(top_srcdir)/include -I$(top_builddir)

bin_PROGRAMS = 	\
	mult-seq 			\
	mult-omp 			\
	mult-tbb 			\
	mult-pthrd	 	\
	mult-kaapi 		\
	elim-seq	 		\
	elim-omp 			\
	elim-kaapi		\
	elim-pthrd		\
	elim-tbb

mult_seq_CFLAGS= $(AM_CFLAGS)
mult_seq_SOURCES= 	\
	mat-mult-seq.c
elim_seq_CFLAGS= $(AM_CFLAGS)
elim_seq_SOURCES= 	\
	mat-elim-seq.c

mult_omp_CFLAGS= $(AM_CFLAGS) -fopenmp
mult_omp_SOURCES=		\
	mat-mult-omp.c
elim_omp_CFLAGS= $(AM_CFLAGS) -fopenmp
elim_omp_SOURCES= 	\
	mat-elim-omp.c

mult_tbb_CPPFLAGS= $(AM_CPPFLAGS) -std=c++0x
mult_tbb_LDADD= -ltbb
mult_tbb_SOURCES= 	\
	mat-mult-tbb.cpp
elim_tbb_CPPFLAGS= $(AM_CPPFLAGS) -std=c++0x
elim_tbb_LDADD= -ltbb
elim_tbb_SOURCES= 	\
	mat-elim-tbb.cpp

mult_pthrd_CFLAGS= $(AM_CFLAGS) -pthread
mult_pthrd_SOURCES= 	\
	mat-mult-pthrd.c
elim_pthrd_CFLAGS= $(AM_CFLAGS) -pthread
elim_pthrd_SOURCES= 	\
	mat-elim-pthrd.c

mult_kaapi_CFLAGS= $(AM_CFLAGS) 
mult_kaapi_LDADD= -lkaapic
mult_kaapi_SOURCES= 	\
	mat-mult-kaapi.c
elim_kaapi_CFLAGS= $(AM_CFLAGS) 
elim_kaapi_LDADD= -lkaapic
elim_kaapi_SOURCES= 	\
	mat-elim-kaapi.c
CLEANFILES=	

DISTCLEANFILES=	\
	mult-seq			\
	mult-tbb			\
	mult-omp			\
	mult-kaapi		\
	mult-pthrd		\
	elim-seq			\
	elim-tbb			\
	elim-omp			\
	elim-kaapi		\
	elim-pthrd
