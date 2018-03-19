#include "std.h"
#include <stdarg.h>
#include "stdio.h"

void std_printf(const char* format, ... ) {
  va_list arglist;
  va_start( arglist, format );
  vprintf( format, arglist );
  va_end(arglist);
}

void err_printf(const char* format, ... ) {
  va_list arglist;
  va_start( arglist, format );
  vfprintf( stderr, format, arglist );
  va_end(arglist);
}
