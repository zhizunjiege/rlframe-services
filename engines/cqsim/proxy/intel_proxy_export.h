
#ifndef INTEL_PROXY_EXPORT_H
#define INTEL_PROXY_EXPORT_H

#ifdef INTEL_PROXY_STATIC_DEFINE
#  define INTEL_PROXY_EXPORT
#  define INTEL_PROXY_NO_EXPORT
#else
#  ifndef INTEL_PROXY_EXPORT
#    ifdef intel_proxy_EXPORTS
        /* We are building this library */
#      define INTEL_PROXY_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define INTEL_PROXY_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef INTEL_PROXY_NO_EXPORT
#    define INTEL_PROXY_NO_EXPORT
#  endif
#endif

#ifndef INTEL_PROXY_DEPRECATED
#  define INTEL_PROXY_DEPRECATED __declspec(deprecated)
#endif

#ifndef INTEL_PROXY_DEPRECATED_EXPORT
#  define INTEL_PROXY_DEPRECATED_EXPORT INTEL_PROXY_EXPORT INTEL_PROXY_DEPRECATED
#endif

#ifndef INTEL_PROXY_DEPRECATED_NO_EXPORT
#  define INTEL_PROXY_DEPRECATED_NO_EXPORT INTEL_PROXY_NO_EXPORT INTEL_PROXY_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef INTEL_PROXY_NO_DEPRECATED
#    define INTEL_PROXY_NO_DEPRECATED
#  endif
#endif

#endif /* INTEL_PROXY_EXPORT_H */
