# ForestKitten 33 Driver Makefile

obj-m := fk33_driver.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	sudo insmod fk33_driver.ko

uninstall:
	sudo rmmod fk33_driver

.PHONY: all clean install uninstall
