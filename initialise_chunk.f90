SUBROUTINE initialise_chunk(chunk)

  USE clover_module

  IMPLICIT NONE

  INTEGER :: chunk

  REAL(KIND=8) :: xmin,ymin,dx,dy

  dx=(grid%xmax-grid%xmin)/float(grid%x_cells)
  dy=(grid%ymax-grid%ymin)/float(grid%y_cells)

  xmin=grid%xmin+dx*float(chunks(chunk)%field%left-1)

  ymin=grid%ymin+dy*float(chunks(chunk)%field%bottom-1)

  CALL initialise_chunk_kernel_ocl(chunks(chunk)%field%x_min,    &
                                   chunks(chunk)%field%x_max,    &
                                   chunks(chunk)%field%y_min,    &
                                   chunks(chunk)%field%y_max,    &
                                   xmin,ymin,dx,dy              )


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

END SUBROUTINE initialise_chunk
