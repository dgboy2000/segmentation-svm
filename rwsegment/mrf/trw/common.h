#ifndef __trw_common_h__
#define __trw_common_h__

/** allocator
 */
// typedef void*(*Allocator)(int, unsigned int*, char*, char*);
typedef void*(*Allocator)(int, unsigned int*, char const*, char const*);

#endif
