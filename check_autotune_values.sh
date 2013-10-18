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

total_wis=`expr ${1} \* ${2}`
check_total 

total_wis=`expr ${3} \* ${4}`
check_total 

total_wis=`expr ${5} \* ${6}`
check_total 

total_wis=`expr ${7} \* ${8}`
check_total 

total_wis=`expr ${9} \* ${10}`
check_total 

total_wis=`expr ${11} \* ${12}`
check_total 
total_wis=`expr ${13} \* ${14}`
check_total 
total_wis=`expr ${15} \* ${16}`
check_total 
total_wis=`expr ${17} \* ${18}`
check_total 
total_wis=`expr ${19} \* ${20}`
check_total 
total_wis=`expr ${21} \* ${22}`
check_total 
total_wis=`expr ${23} \* ${24}`
check_total 
total_wis=`expr ${25} \* ${26}`
check_total 
total_wis=`expr ${27} \* ${28}`
check_total 
total_wis=`expr ${29} \* ${30}`
check_total 
total_wis=`expr ${31} \* ${32}`
check_total 
total_wis=`expr ${33} \* ${34}`
check_total 
total_wis=`expr ${35} \* ${36}`
check_total 
total_wis=`expr ${37} \* ${38}`
check_total 
total_wis=`expr ${39} \* ${40}`
check_total 
total_wis=`expr ${41} \* ${42}`
check_total 
total_wis=`expr ${43} \* ${44}`
check_total 
total_wis=`expr ${45} \* ${46}`
check_total 
total_wis=`expr ${47} \* ${48}`
check_total 
total_wis=`expr ${49} \* ${50}`
check_total 
total_wis=`expr ${51} \* ${52}`
check_total 
total_wis=`expr ${53} \* ${54}`
check_total 

echo 0
exit 0


