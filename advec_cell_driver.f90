!Crown Copyright 2012 AWE.
!
! This file is part of CloverLeaf.
!
! CloverLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! CloverLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! CloverLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Cell centred advection driver.
!>  @author Wayne Gaudin
!>  @details Invokes the user selected advection kernel.

MODULE  advec_cell_driver_module

CONTAINS

SUBROUTINE advec_cell_driver(chunk,sweep_number,dir)

  USE clover_module

  IMPLICIT NONE

  INTEGER :: chunk,sweep_number,dir,vector

  IF(chunks(chunk)%task.EQ.parallel%task) THEN

    CALL advec_cell_kernel_ocl(chunks(chunk)%field%x_min,          &
                        chunks(chunk)%field%x_max,                 &
                        chunks(chunk)%field%y_min,                 &
                        chunks(chunk)%field%y_max,                 &
                        dir,                                       &
                        sweep_number                               )

  ENDIF

END SUBROUTINE advec_cell_driver

END MODULE  advec_cell_driver_module

