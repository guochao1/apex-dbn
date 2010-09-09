#!/usr/bin/python

from os import sys
from os import system

def mkmaps( level, n ):
    for i in xrange( n ):
        src = '  models/%02d.%04d.model' % (level,i)
        dst = '  maps/%02d.%04d.bmp' % (level,i)
        cmd = 'crbm_model_drawer ' + src + dst

        for k in xrange( 3 , len( sys.argv )):    
            cmd = cmd + ' ' + sys.argv[k]           
        print cmd

        system( cmd )

        dst = '  dump/%02d.%04d.txt' % ( level , i )
        cmd = 'crbm_model_dumper ' + src + dst
        print cmd
        system( cmd )

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage:<num> <level> <method> [...]'
        exit(-1)
    mkmaps( int( sys.argv[2]), int(sys.argv[1]) )

