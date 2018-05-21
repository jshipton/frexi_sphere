#! /bin/bash

#
# Use this bash script to setup environment variables to directly work with this repository
#
# echo "     |    $ source ./env_vars.sh"
#


if [ "#$0" != "#-bash" ]; then
        if [ "`basename $0`" == "env_vars.sh" ]; then

		echo "ERROR|"
		echo "ERROR| >>> $0"
		echo "ERROR| THIS SCRIPT MAY NOT BE EXECUTED, BUT INCLUDED IN THE ENVIRONMENT VARIABLES!"
		echo "ERROR| Use e.g. "
		echo "     |"
		echo "     |    $ source ./env_vars.sh"
		echo "     |"
		return

	fi
fi

BACKDIR="$PWD"

test "x${PWD##*/}" = "xlocal_software" && cd ../

SCRIPTDIR="`pwd`/"

export PYSDC_ROOT="`pwd`"


if [ ! -d "$SCRIPTDIR" ]; then
        echo
        echo "ERROR| Execute this script only from the PYSDC root directory"
        echo "     |   $ source /env_vars.sh"
	echo "     or"
        echo "     |   $ . ./env_vars.sh"
        echo
        return
fi


export PS1="[FREXI] $PS1"

export PYTHONPATH="$PYSDC_ROOT:$PYTHONPATH"
