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

!>  @brief Momentum advection driver
!>  @author Wayne Gaudin
!>  @details Invokes the user specified momentum advection kernel.

MODULE advec_mom_driver_module

CONTAINS

SUBROUTINE advec_mom_driver(chunk,which_vel,direction,sweep_number)

  USE clover_module

  IMPLICIT NONE

  INTEGER :: chunk,which_vel,direction,sweep_number,vector

  IF(chunks(chunk)%task.EQ.parallel%task) THEN

    IF (use_vector_loops) THEN
        vector=1
    ELSE
        vector=0
    ENDIF
    CALL advec_mom_kernel_ocl(chunks(chunk)%field%x_min,          &
                              chunks(chunk)%field%x_max,          &
                              chunks(chunk)%field%y_min,          &
                              chunks(chunk)%field%y_max,          &
                              which_vel,                          &
                              sweep_number,                       &
                              direction,                          &
                              vector                              )

    CALL ocl_read_back_all_buffers(chunks(chunk)%field%density0,    &
                                   chunks(chunk)%field%density1,    &
                                   chunks(chunk)%field%energy0,     &
                                   chunks(chunk)%field%energy1,     &
                                   chunks(chunk)%field%pressure,    &
                                   chunks(chunk)%field%viscosity,   &
                                   chunks(chunk)%field%soundspeed,  &
                                   chunks(chunk)%field%xvel0,       &
                                   chunks(chunk)%field%xvel1,       &
                                   chunks(chunk)%field%yvel0,       &
                                   chunks(chunk)%field%yvel1,       &
                                   chunks(chunk)%field%vol_flux_x,  &
                                   chunks(chunk)%field%mass_flux_x, &
                                   chunks(chunk)%field%vol_flux_y,  &
                                   chunks(chunk)%field%mass_flux_y)

  ENDIF

END SUBROUTINE advec_mom_driver

END MODULE advec_mom_driver_module
