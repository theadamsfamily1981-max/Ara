# User-space API Library Makefile

LIB_NAME = libsnn
LIB_MAJOR = 1
LIB_MINOR = 0
LIB_PATCH = 0
LIB_SO = $(LIB_NAME).so.$(LIB_MAJOR).$(LIB_MINOR).$(LIB_PATCH)
LIB_SONAME = $(LIB_NAME).so.$(LIB_MAJOR)
LIB_LINK = $(LIB_NAME).so

CC = gcc
CFLAGS = -Wall -Werror -fPIC -O2 -I../include
LDFLAGS = -shared -Wl,-soname,$(LIB_SONAME)

SOURCES = snn_api.c
OBJECTS = $(SOURCES:.c=.o)

PREFIX ?= /usr/local
LIBDIR = $(PREFIX)/lib
INCLUDEDIR = $(PREFIX)/include

all: $(LIB_SO)

$(LIB_SO): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^
	ln -sf $(LIB_SO) $(LIB_SONAME)
	ln -sf $(LIB_SONAME) $(LIB_LINK)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIB_SO) $(LIB_SONAME) $(LIB_LINK)

install: $(LIB_SO)
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)/snn_kernel
	install -m 755 $(LIB_SO) $(DESTDIR)$(LIBDIR)
	ln -sf $(LIB_SO) $(DESTDIR)$(LIBDIR)/$(LIB_SONAME)
	ln -sf $(LIB_SONAME) $(DESTDIR)$(LIBDIR)/$(LIB_LINK)
	install -m 644 ../include/snn_kernel/*.h $(DESTDIR)$(INCLUDEDIR)/snn_kernel/
	ldconfig

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SO)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SONAME)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_LINK)
	rm -rf $(DESTDIR)$(INCLUDEDIR)/snn_kernel
	ldconfig

.PHONY: all clean install uninstall
