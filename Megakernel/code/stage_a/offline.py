"""Reject Python IPv4/IPv6 connects and DNS during experimental entry points."""
import socket
import sys
import os
import shlex

DISABLED_EXECUTABLES={"ncu","nv-nsight-cu-cli","nv-nsight-cu","wget","curl"}


def activate():
    def audit(event,args):
        if event=="subprocess.Popen":
            executable,argv=args[:2]
            if executable is None:
                executable=shlex.split(argv,posix=False)[0].strip('"') if isinstance(argv,str) else argv[0]
            name=os.path.basename(os.fsdecode(executable)).lower()
            if name.removesuffix(".exe") in DISABLED_EXECUTABLES:
                raise RuntimeError("Stage A offline policy: disabled external executable")
        if event=="socket.getaddrinfo":
            raise RuntimeError("Stage A offline policy: DNS disabled")
        if event in ("socket.connect","socket.sendto","socket.sendmsg") and args[0].family in (socket.AF_INET,socket.AF_INET6):
            raise RuntimeError("Stage A offline policy: network disabled")
    sys.addaudithook(audit)
