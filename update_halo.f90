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

!>  @brief Driver for the halo updates
!>  @author Wayne Gaudin
!>  @details Invokes the kernels for the internal and external halo cells for
!>  the fields specified.

MODULE update_halo_module

CONTAINS

SUBROUTINE update_halo(fields,depth)

  USE clover_module

  IMPLICIT NONE

  INTEGER :: c,fields(NUM_FIELDS),depth

  CALL clover_exchange(fields,depth)

  DO c=1,number_of_chunks

    IF(chunks(c)%task.EQ.parallel%task) THEN

      CALL update_halo_kernel_ocl(chunks(c)%field%x_min,          &
                             chunks(c)%field%x_max,          &
                             chunks(c)%field%y_min,          &
                             chunks(c)%field%y_max,          &
                             chunks(c)%field%left,           &
                             chunks(c)%field%bottom,         &
                             chunks(c)%field%right,          &
                             chunks(c)%field%top,            &
                             chunks(c)%field%left_boundary,  &
                             chunks(c)%field%bottom_boundary,&
                             chunks(c)%field%right_boundary, &
                             chunks(c)%field%top_boundary,   &
                             chunks(c)%chunk_neighbours,     &
                             fields,                         &
                             depth                           )

    ENDIF

  ENDDO

END SUBROUTINE update_halo

END MODULE update_halo_module
