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

!>  @brief Ideal gas kernel driver
!>  @author Wayne Gaudin
!>  @details Invokes the user specified kernel for the ideal gas equation of
!>  state using the specified time level data.

MODULE ideal_gas_module

CONTAINS

SUBROUTINE ideal_gas(chunk,predict)

  USE clover_module

  IMPLICIT NONE

  INTEGER :: chunk

  LOGICAl :: predict
  INTEGER :: prdct

  IF(chunks(chunk)%task .EQ. parallel%task) THEN

    IF(predict) THEN
          prdct=0
    ELSE
          prdct=1
    ENDIF

    CALL ideal_gas_kernel_ocl(chunks(chunk)%field%x_min,  &
                        chunks(chunk)%field%x_max,      &
                        chunks(chunk)%field%y_min,      &
                        chunks(chunk)%field%y_max,      &
                        prdct)

    !CALL ocl_read_back_all_buffers(chunks(chunk)%field%density0,    &
    !                               chunks(chunk)%field%density1,    &
    !                               chunks(chunk)%field%energy0,     &
    !                               chunks(chunk)%field%energy1,     &
    !                               chunks(chunk)%field%pressure,    &
    !                               chunks(chunk)%field%viscosity,   &
    !                               chunks(chunk)%field%soundspeed,  &
    !                               chunks(chunk)%field%xvel0,       &
    !                               chunks(chunk)%field%xvel1,       &
    !                               chunks(chunk)%field%yvel0,       &
    !                               chunks(chunk)%field%yvel1,       &
    !                               chunks(chunk)%field%vol_flux_x,  &
    !                               chunks(chunk)%field%mass_flux_x, &
    !                               chunks(chunk)%field%vol_flux_y,  &
    !                               chunks(chunk)%field%mass_flux_y)

  ENDIF

END SUBROUTINE ideal_gas

END MODULE ideal_gas_module
