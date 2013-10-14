#!/bin/bash

function check_total {
    case $total_wis in
        32) ;;
        64) ;;
        128) ;;
        256) ;;
        512) ;;
        1024) ;;
        *) echo 1; exit 1 ;;
    esac
}

total_wis=`expr $1 \* $2`
check_total 
echo 0
exit 0


