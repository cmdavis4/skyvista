from carlee_tools import spacing


class FPS:

    def __init__(
        self,
        fps=None,
        data_seconds_per_wall_second=None,
        wall_seconds_per_data_second=None,
        total_wall_seconds=None,
        raise_if_not_evenly_spaced=True,
    ) -> None:
        if not any(
            [
                fps,
                data_seconds_per_wall_second,
                wall_seconds_per_data_second,
                total_wall_seconds,
            ]
        ):
            raise ValueError("Must pass one argument to FPS")
        self.passed_fps = fps
        self.data_seconds_per_wall_second = data_seconds_per_wall_second
        self.wall_seconds_per_data_second = wall_seconds_per_data_second
        self.total_wall_seconds = total_wall_seconds
        self.raise_if_not_evenly_spaced = raise_if_not_evenly_spaced

    def to_fps(self, data_times=None):
        """
        Convert FPS specification to actual frames per second.

        Parameters
        ----------
        data_times : array-like, optional
            data time values (required unless fps is directly specified)

        Returns
        -------
        float
            Frames per second for the animation
        """
        if self.passed_fps:
            return self.passed_fps
        elif data_times is None:
            raise ValueError("Must pass data times if not passing an fps directly")
        else:
            # Get the timestep between frames (in seconds)
            dt = (
                spacing(
                    data_times,
                    raise_if_not_evenly_spaced=self.raise_if_not_evenly_spaced,
                )
                .astype("timedelta64[s]")
                .astype(float)
                .item()
            )
            n_frames = len(data_times)

            if self.data_seconds_per_wall_second:
                # data seconds per wall second is a speedup factor
                # (how many data seconds pass per wall clock second)
                # If dt is the data timestep, and we show every frame:
                # wall_seconds_per_frame = dt / data_seconds_per_wall_second
                # fps = 1 / wall_seconds_per_frame = data_seconds_per_wall_second / dt
                fps = self.data_seconds_per_wall_second / dt

            elif self.wall_seconds_per_data_second:
                # Wall seconds per data second is a speedup factor
                # If dt is the data timestep, and we show every frame:
                # wall_seconds_per_frame = dt * wall_seconds_per_data_second
                # fps = 1 / wall_seconds_per_frame
                wall_seconds_per_frame = dt * self.wall_seconds_per_data_second
                fps = 1.0 / wall_seconds_per_frame

            elif self.total_wall_seconds:
                # Total wall time tells us the total animation duration
                # fps = number_of_frames / total_wall_time
                fps = n_frames / self.total_wall_seconds

            else:
                raise ValueError(
                    "FPS object must have one of: fps,"
                    " data_seconds_per_wall_second,"
                    " wall_seconds_per_data_second, or total_wall_time"
                )

            return fps
