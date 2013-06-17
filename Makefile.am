# Copyright 2012 Christian Eder
# 
# This file is part of pLA, licensed under the GNU General Public
# License version 3. See COPYING for more information.

INCLUDES=-I$(top_srcdir) -I$(top_srcdir)/include -I$(top_builddir)

pkgincludesubdir = $(includedir)/pla
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = pla.pc

SUBDIRS= . src m4
OBJEXT=".lo .o"
ACLOCAL_AMFLAGS=-I m4

bin_SCRIPTS=pla-config

EXTRA_DIST=PROJECTS

CLEANFILES=				\
	tags						\
	obj/*						\
	*~



distclean-local:
	rm -rf autom4te.cache
	rm -rf obj

DISTCLEANFILES= 												\
	_configs.sed													\
	config.sub														\
	config.guess													\
	configure.in													\
	configure															\
	ltmain.sh															\
	Makefile.in														\
	pla-config pla-config~								\
	include/pla-config.h*									\
	include/stamp-h1a         						\
	obj/*																	\
	*~

ctags:
	ctags -R *

all-am:
