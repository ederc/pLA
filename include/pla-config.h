#ifndef _INCLUDE_PLA_CONFIG_H
#define _INCLUDE_PLA_CONFIG_H 1
 
/* include/pla-config.h. Generated automatically at end of configure. */
/* include/pla-config.h.  Generated from pla-config.h.in by configure.  */
/* include/pla-config.h.in.  Generated from configure.ac by autoheader.  */

/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* it is a macro in order to print out the correct file and line */
#ifndef __PLA_ASSERT
#define __PLA_ASSERT(x) xAssert(x,__FILE__,__LINE__)
#endif

/* "atomic(9) operations as provided on FreeBSD" */
/* #undef ATOMIC9 */

/* bitsize of char */
#ifndef __PLA_BIT_SIZEOF_CHAR
#define __PLA_BIT_SIZEOF_CHAR 8
#endif

/* depends on bitsize of char and size of long */
#ifndef __PLA_BIT_SIZEOF_LONG
#define __PLA_BIT_SIZEOF_LONG (__PLA_BIT_SIZEOF_CHAR * __PLA_SIZEOF_LONG)
#endif

/* "not silencing irrelevant compiler warnings by default */
/* #undef CC_SILENCE */

/* cache line size in bytes ( default 64 byte ) */
#ifndef __PLA_CPU_CACHE_LINE
#define __PLA_CPU_CACHE_LINE 
#endif

/* L1 cache size */
#ifndef __PLA_CPU_L1_CACHE
#define __PLA_CPU_L1_CACHE 
#endif

/* L2 cache size */
#ifndef __PLA_CPU_L2_CACHE
#define __PLA_CPU_L2_CACHE 
#endif

/* "spinwait macro for pLA" */
#ifndef __PLA_CPU_SPINWAIT
#define __PLA_CPU_SPINWAIT __asm__ volatile("pause")
#endif

/* dss available */
/* #undef DSS */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define to a macro mangling the given Fortan function name */
#ifndef __PLA_F77_FUNC
#define __PLA_F77_FUNC(name) name ## _
#endif

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* junk / zero filling enabled */
#ifndef __PLA_FILL
#define __PLA_FILL 1
#endif

/* "__sync_{addsub}_and_fetch() " */
/* #undef FORCE_SYNC_COMPARE_AND_SWAP_4 */

/* "__sync_{addsub}_and_fetch() " */
/* #undef FORCE_SYNC_COMPARE_AND_SWAP_8 */

/* Define if you have a ATLAS library. */
/* #undef HAVE_ATLAS */

/* "check if __attribute__ syntax is supported" */
#ifndef __PLA_HAVE_ATTR
#define __PLA_HAVE_ATTR 
#endif

/* Support AVX (Advanced Vector Extensions) instructions */
/* #undef HAVE_AVX */

/* Define that architecture uses big endian storage */
/* #undef HAVE_BIG_ENDIAN */

/* Define if you have a BLAS library. */
#ifndef __PLA_HAVE_BLAS
#define __PLA_HAVE_BLAS /**/
#endif

/* Define to 1 if you have the <dlfcn.h> header file. */
/* #undef HAVE_DLFCN_H */

/* Define to 1 if you have the `getpagesize' function. */
#ifndef __PLA_HAVE_GETPAGESIZE
#define __PLA_HAVE_GETPAGESIZE 1
#endif

/* "hwloc available." */
#ifndef __PLA_HAVE_HWLOC
#define __PLA_HAVE_HWLOC /**/
#endif

/* Define if you have an Intel MKL library. */
/* #undef HAVE_INTEL_MKL */

/* "Intel TBB available." */
#ifndef __PLA_HAVE_INTEL_TBB
#define __PLA_HAVE_INTEL_TBB /**/
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef __PLA_HAVE_INTTYPES_H
#define __PLA_HAVE_INTTYPES_H 1
#endif

/* "KAAPI available." */
#ifndef __PLA_HAVE_KAAPI
#define __PLA_HAVE_KAAPI /**/
#endif

/* "KAAPIC available." */
#ifndef __PLA_HAVE_KAAPIC
#define __PLA_HAVE_KAAPIC /**/
#endif

/* "KAAPI++ available." */
#ifndef __PLA_HAVE_KAAPIPP
#define __PLA_HAVE_KAAPIPP /**/
#endif

/* Define if you have LAPACK library. */
#ifndef __PLA_HAVE_LAPACK
#define __PLA_HAVE_LAPACK 1
#endif

/* Define that architecture uses little endian storage */
#ifndef __PLA_HAVE_LITTLE_ENDIAN
#define __PLA_HAVE_LITTLE_ENDIAN 1
#endif

/* Define to 1 if you have the <memory.h> header file. */
#ifndef __PLA_HAVE_MEMORY_H
#define __PLA_HAVE_MEMORY_H 1
#endif

/* Define to 1 if you have a working `mmap' system call. */
#ifndef __PLA_HAVE_MMAP
#define __PLA_HAVE_MMAP 1
#endif

/* Support MMX instructions */
/* #undef HAVE_MMX */

/* Define if you have an OpenBLAS library. */
/* #undef HAVE_OPENBLAS */

/* Define if OpenMP is enabled */
#ifndef __PLA_HAVE_OPENMP
#define __PLA_HAVE_OPENMP 1
#endif

/* Define to 1 if you have the <pthread.h> header file. */
#ifndef __PLA_HAVE_PTHREAD_H
#define __PLA_HAVE_PTHREAD_H 1
#endif

/* sbrk available ? */
#ifndef __PLA_HAVE_SBRK
#define __PLA_HAVE_SBRK 1
#endif

/* Support SSE (Streaming SIMD Extensions) instructions */
/* #undef HAVE_SSE */

/* Support SSE2 (Streaming SIMD Extensions 2) instructions */
/* #undef HAVE_SSE2 */

/* Support SSE3 (Streaming SIMD Extensions 3) instructions */
/* #undef HAVE_SSE3 */

/* Support SSE4.1 (Streaming SIMD Extensions 4.1) instructions */
/* #undef HAVE_SSE41 */

/* Support SSE4.2 (Streaming SIMD Extensions 4.2) instructions */
/* #undef HAVE_SSE42 */

/* Support SSSE3 (Supplemental Streaming SIMD Extensions 3) instructions */
/* #undef HAVE_SSSE3 */

/* Define to 1 if you have the <stdint.h> header file. */
#ifndef __PLA_HAVE_STDINT_H
#define __PLA_HAVE_STDINT_H 1
#endif

/* Define to 1 if you have the <stdlib.h> header file. */
#ifndef __PLA_HAVE_STDLIB_H
#define __PLA_HAVE_STDLIB_H 1
#endif

/* Define to 1 if you have the <strings.h> header file. */
#ifndef __PLA_HAVE_STRINGS_H
#define __PLA_HAVE_STRINGS_H 1
#endif

/* Define to 1 if you have the <string.h> header file. */
#ifndef __PLA_HAVE_STRING_H
#define __PLA_HAVE_STRING_H 1
#endif

/* Define to 1 if you have the <sys/param.h> header file. */
#ifndef __PLA_HAVE_SYS_PARAM_H
#define __PLA_HAVE_SYS_PARAM_H 1
#endif

/* Define to 1 if you have the <sys/stat.h> header file. */
#ifndef __PLA_HAVE_SYS_STAT_H
#define __PLA_HAVE_SYS_STAT_H 1
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#ifndef __PLA_HAVE_SYS_TYPES_H
#define __PLA_HAVE_SYS_TYPES_H 1
#endif

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef __PLA_HAVE_UNISTD_H
#define __PLA_HAVE_UNISTD_H 1
#endif

/* Depending on LOG_BIT_SIZEOF_LONG */
#ifndef __PLA_INDEX_PAGE_SHIFT
#define __PLA_INDEX_PAGE_SHIFT (__PLA_LOG_BIT_SIZEOF_LONG + __PLA_LOG_BIT_SIZEOF_SYSTEM_PAGE)
#endif

/* Canonical 16-bit data type */
#ifndef __PLA_INT16
#define __PLA_INT16 short
#endif

/* Canonical 32-bit data type */
#ifndef __PLA_INT32
#define __PLA_INT32 int
#endif

/* Canonical 64-bit data type */
#ifndef __PLA_INT64
#define __PLA_INT64 long
#endif

/* Canonical 8-bit data type */
#ifndef __PLA_INT8
#define __PLA_INT8 char
#endif

/* "darwin related configuratioin" */
/* #undef IVSALLOC */

/* "Macro for lazy locking: Lock only if multi-threaded" */
/* #undef LAZY_LOCK */

/* Depending on SIZEOF_LONG */
#ifndef __PLA_LOG_BIT_SIZEOF_LONG
#define __PLA_LOG_BIT_SIZEOF_LONG 6
#endif

/* Log bit size of system page */
#ifndef __PLA_LOG_BIT_SIZEOF_SYSTEM_PAGE
#define __PLA_LOG_BIT_SIZEOF_SYSTEM_PAGE 12
#endif

/* log cache line size */
#ifndef __PLA_LOG_CPU_CACHE_LINE
#define __PLA_LOG_CPU_CACHE_LINE 
#endif

/* log bitsize of alignment of memory allocated by pla */
#ifndef __PLA_LOG_SIZEOF_ALIGNMENT
#define __PLA_LOG_SIZEOF_ALIGNMENT 3
#endif

/* Depending on SIZEOF_LONG */
#ifndef __PLA_LOG_SIZEOF_LONG
#define __PLA_LOG_SIZEOF_LONG 3
#endif

/* maximum of two comparable values */
#ifndef __PLA_MAX
#define __PLA_MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

/* depending on the chosen size classes and their subdivision */
#ifndef __PLA_MAX_BIN_INDEX
#define __PLA_MAX_BIN_INDEX 22
#endif

/* Depending on SIZEOF_LONG */
#ifndef __PLA_MAX_SMALL_BLOCK_SIZE
#define __PLA_MAX_SMALL_BLOCK_SIZE 1008
#endif

/* minimum of two comparable values */
#ifndef __PLA_MIN
#define __PLA_MIN(a,b) ((a) > (b) ? (b) : (a))
#endif

/* default minimal value of the number of pages allocated for a new region */
#ifndef __PLA_MIN_NUMBER_PAGES_PER_REGION
#define __PLA_MIN_NUMBER_PAGES_PER_REGION 512
#endif

/* "checks if BSD-specific _pthread_mutex_init_calloc_cb() exists" */
/* #undef MUTEX_INIT_CB */

/* Define to 1 if assertions should be disabled. */
/* #undef NDEBUG */

/* Jump to next entry */
#ifndef __PLA_NEXT
#define __PLA_NEXT(addr) *((void**) addr)
#endif

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* "atomic(3) operations as provided on Darwin / Mac OS X" */
/* #undef OSATOMIC */

/* "spinlock(3) operations as provided on Darwin / Mac OS X" */
/* #undef OSSPIN */

/* Name of package */
#ifndef __PLA_PACKAGE
#define __PLA_PACKAGE "pla"
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef __PLA_PACKAGE_BUGREPORT
#define __PLA_PACKAGE_BUGREPORT "ederc@mathematik.uni-kl.de"
#endif

/* Define to the full name of this package. */
#ifndef __PLA_PACKAGE_NAME
#define __PLA_PACKAGE_NAME "pLA"
#endif

/* Define to the full name and version of this package. */
#ifndef __PLA_PACKAGE_STRING
#define __PLA_PACKAGE_STRING "pLA 0.0.1"
#endif

/* Define to the one symbol short name of this package. */
#ifndef __PLA_PACKAGE_TARNAME
#define __PLA_PACKAGE_TARNAME "pla"
#endif

/* Define to the home page for this package. */
#ifndef __PLA_PACKAGE_URL
#define __PLA_PACKAGE_URL ""
#endif

/* Define to the version of this package. */
#ifndef __PLA_PACKAGE_VERSION
#define __PLA_PACKAGE_VERSION "0.0.1"
#endif

/* Depending on SIZEOF_LONG */
#ifndef __PLA_PAGES_PER_REGION
#define __PLA_PAGES_PER_REGION 512
#endif

/* "" */
#ifndef __PLA_PURGE_MADVISE_DONTNEED
#define __PLA_PURGE_MADVISE_DONTNEED 
#endif

/* "" */
/* #undef PURGE_MADVISE_FREE */

/* bitsize of alignment of memory allocated by pla */
#ifndef __PLA_SIZEOF_ALIGNMENT
#define __PLA_SIZEOF_ALIGNMENT 8
#endif

/* bitsize of alignment of memory allocated by pla */
#ifndef __PLA_SIZEOF_ALIGNMENT_MINUS_ONE
#define __PLA_SIZEOF_ALIGNMENT_MINUS_ONE 7
#endif

/* The size of `char', as computed by sizeof. */
#ifndef __PLA_SIZEOF_CHAR
#define __PLA_SIZEOF_CHAR 1
#endif

/* Depending on LOG_BIT_SIZEOF_LONG */
#ifndef __PLA_SIZEOF_INDEX_PAGE_MINUS_ONE
#define __PLA_SIZEOF_INDEX_PAGE_MINUS_ONE ((__PLA_SIZEOF_SYSTEM_PAGE << __PLA_LOG_BIT_SIZEOF_LONG) - 1)
#endif

/* The size of `int', as computed by sizeof. */
#ifndef __PLA_SIZEOF_INT
#define __PLA_SIZEOF_INT 4
#endif

/* The size of `long', as computed by sizeof. */
#ifndef __PLA_SIZEOF_LONG
#define __PLA_SIZEOF_LONG 8
#endif

/* The size of `long long', as computed by sizeof. */
#ifndef __PLA_SIZEOF_LONG_LONG
#define __PLA_SIZEOF_LONG_LONG 8
#endif

/* Depending on SIZEOF_SYSTEM_PAGE and SIZEOF_PAGE_HEADER */
#ifndef __PLA_SIZEOF_PAGE
#define __PLA_SIZEOF_PAGE (__PLA_SIZEOF_SYSTEM_PAGE - __PLA_SIZEOF_PAGE_HEADER)
#endif

/* Depending on SIZEOF_LONG and SIZEOF_VOIDP */
#ifndef __PLA_SIZEOF_PAGE_HEADER
#define __PLA_SIZEOF_PAGE_HEADER (5*__PLA_SIZEOF_VOIDP + __PLA_SIZEOF_LONG)
#endif

/* The size of `short', as computed by sizeof. */
#ifndef __PLA_SIZEOF_SHORT
#define __PLA_SIZEOF_SHORT 2
#endif

/* size of strict alignment of memory allocated by pla */
#ifndef __PLA_SIZEOF_STRICT_ALIGNMENT
#define __PLA_SIZEOF_STRICT_ALIGNMENT 8
#endif

/* Size of system page */
#ifndef __PLA_SIZEOF_SYSTEM_PAGE
#define __PLA_SIZEOF_SYSTEM_PAGE 4096
#endif

/* The size of `void*', as computed by sizeof. */
#ifndef __PLA_SIZEOF_VOIDP
#define __PLA_SIZEOF_VOIDP 8
#endif

/* depending on sizeof(void*) */
#ifndef __PLA_SIZEOF_VOIDP_MINUS_ONE
#define __PLA_SIZEOF_VOIDP_MINUS_ONE (__PLA_SIZEOF_VOIDP - 1)
#endif

/* The size of `__int64', as computed by sizeof. */
#ifndef __PLA_SIZEOF___INT64
#define __PLA_SIZEOF___INT64 0
#endif

/* The size of `__uint128_t', as computed by sizeof. */
#ifndef __PLA_SIZEOF___UINT128_T
#define __PLA_SIZEOF___UINT128_T 16
#endif

/* The size of `__uint256_t', as computed by sizeof. */
#ifndef __PLA_SIZEOF___UINT256_T
#define __PLA_SIZEOF___UINT256_T 0
#endif

/* Define to 1 if you have the ANSI C header files. */
#ifndef __PLA_STDC_HEADERS
#define __PLA_STDC_HEADERS 1
#endif

/* macro stringification mainly used by xassert */
#ifndef __PLA_STRINGIFICATION
#define __PLA_STRINGIFICATION(x) #x
#endif

/* "" */
#ifndef __PLA_THREADED_INIT
#define __PLA_THREADED_INIT 
#endif

/* "thread cleanup for pLA" */
/* #undef THREAD_CLEANUP */

/* "Macro for using thread-local storage in pLA." */
#ifndef __PLA_TLS
#define __PLA_TLS 
#endif

/* "model of tls attribute support ( clang 3.0 still lacks support )" */
#ifndef __PLA_TLS_MODEL
#define __PLA_TLS_MODEL __attribute__((tls_model("initial-exec")))
#endif

/* Canonical 128-bit data type */
#ifndef __PLA_UINT128
#define __PLA_UINT128 __uint128_t
#endif

/* Canonical 256-bit data type */
/* #undef UINT256 */

/* valloc does not use mmap */
#ifndef __PLA_VALLOC
#define __PLA_VALLOC xVallocMmap
#endif

/* Version number of package */
#ifndef __PLA_VERSION
#define __PLA_VERSION "0.0.1"
#endif

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* "darwin related configuratioin" */
/* #undef ZONE */

/* "szone version in darwin: there was a jump from 3 to 6 between OS X 10.5.x
   and 10.6" */
/* #undef ZONE_VERSION */
 
/* once: _INCLUDE_PLA_CONFIG_H */
#endif
