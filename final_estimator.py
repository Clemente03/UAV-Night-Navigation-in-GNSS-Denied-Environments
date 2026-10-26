"""

final_estimator.py — star attitude + local vertical -> geodetic latitude and longitude

    from final_estimator import estimator
    loc = estimator.estimate_lat_lon_analytical(q_engine_result=q, utc_dt=utc,
              imu_yaw=yaw, imu_pitch=pitch, imu_roll=roll,
              r_offset_matrix=np.load("R_offset.npy"), dut1=0,
              v_zenith_body=acc_mean)                  #None -> vertical from Euler angles

estimate_lat_lon_analytical() takes the attitude from stilter and the vertical from the IMU
and returns {"lat_deg", "lon_deg", "success"}. The chain, in order:
    R_cirs_to_cam  - the stilter quaternion (x, y, z, w) as a rotation, CIRS -> camera
    v_zenith_imu   - the local vertical in the IMU frame, from one of two sources:
                       v_zenith_body  the accelerometer vector, if given (normalised)
                       Euler angles   otherwise: ZYX, pitch restored from the driver's
                                      90 - theta, zenith along -z of the navigation frame
    v_zenith_cam   - R_offset applied to v_zenith_imu
    v_zenith_cirs  - back into CIRS by the transpose of R_cirs_to_cam. The catalogue
                     vectors were already CIRS, so ICRS -> CIRS is NOT applied again
    aberration     - annual aberration removed to first order (ERFA epv00): the zenith
                     read off the apparent sky sits up to 20.5" toward the apex
latitude = declination of the zenith; longitude = its RA minus the Earth Rotation Angle
(ERFA era00 with dut1, 0 used throughout), wrapped to +-180 deg.

helpers: jd_from_datetime, earth_rotation_angle, get_icrs_to_cirs_matrix (IAU 2006/2000A,
also used by stilter), quaternion_to_matrix, remove_aberration, convert_coordinates

The caller passes UTC: correct the board clock first (it ran on WEST, -3600 s;
one hour of clock error is 1302 km)

Pedro Clemente, 2026-10-03

"""

import numpy as np
import math
from datetime import timezone
import erfa

class estimator:
    #values from imu
    """
    @staticmethod
    def get_zenith_vector_from_imu(pitch_deg, roll_deg):
        p = np.radians(pitch_deg)
        r = np.radians(roll_deg)
        #unit zenith vector in camera frame
        cp = np.cos(p)
        return np.array([np.sin(r) * cp, -np.sin(p), np.cos(r) * cp])
    """
    
    #ERA from UTC datetime
    @staticmethod
    def jd_from_datetime(dt):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() / 86400.0 + 2440587.5
    
    @staticmethod
    def earth_rotation_angle(jd_utc, dut1=0.0):
        utc1 = 2451545.0
        utc2 = jd_utc - 2451545.0

        ut11, ut12 = erfa.utcut1(utc1, utc2, dut1)

        return erfa.era00(ut11, ut12)

    @staticmethod
    def get_icrs_to_cirs_matrix(jd_tt):
        #IAU 2006/2000A
        return erfa.c2i06a(2451545.0, jd_tt - 2451545.0)

    @staticmethod
    def quaternion_to_matrix(q):
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

    @staticmethod
    def remove_aberration(v_apparent, jd_utc):
        """Annual aberration, first order: apparent direction -> geometric.
        Tetra3 fits the aberrated sky, so the zenith it returns is displaced by
        up to 20.5" toward the apex of Earth's motion; undo it here."""
        jd_tt = jd_utc + 69.2 / 86400.0
        _, pvb = erfa.epv00(2451545.0, jd_tt - 2451545.0)         #barycentric [pos, vel], AU and AU/day
        beta = np.asarray(pvb[1]) / 173.1446326846693             #Earth velocity as a fraction of c
        v = np.asarray(v_apparent, dtype=float)
        v = v + beta - np.dot(v, beta) * v                        #inverse of p_app = p + beta - (p.beta) p
        return v / np.linalg.norm(v)

    @classmethod
    def estimate_lat_lon_analytical(cls, q_engine_result, utc_dt, imu_yaw, imu_pitch, imu_roll, r_offset_matrix, dut1=0.0, v_zenith_body=None):
        from scipy.spatial.transform import Rotation as R

        q = q_engine_result["quaternion"]
        R_cirs_to_cam = cls.quaternion_to_matrix(q)

        if v_zenith_body is not None:
            v_zenith_imu = np.asarray(v_zenith_body, dtype=float)
            v_zenith_imu = v_zenith_imu / np.linalg.norm(v_zenith_imu)
        else:
            R_imu = R.from_euler('ZYX', [imu_yaw, 90.0 - imu_pitch, imu_roll], degrees=True)
            v_zenith_imu = R_imu.inv().apply([0.0, 0.0, -1.0])

        R_offset = R.from_matrix(r_offset_matrix)
        v_zenith_cam = R_offset.apply(v_zenith_imu)

        v_zenith_cirs = R_cirs_to_cam.T @ v_zenith_cam
        v_zenith_cirs /= np.linalg.norm(v_zenith_cirs)

        jd_utc = cls.jd_from_datetime(utc_dt)
        v_zenith_cirs = cls.remove_aberration(v_zenith_cirs, jd_utc)
        
        #jd_tt = jd_utc + 69.2 / 86400.0

        #R_icrs_to_cirs = cls.get_icrs_to_cirs_matrix(jd_tt)
        #v_zenith_cirs = R_icrs_to_cirs @ v_zenith_cirs
        #v_zenith_cirs /= np.linalg.norm(v_zenith_cirs)

        zenith_ra_rad = np.arctan2(v_zenith_cirs[1], v_zenith_cirs[0])
        zenith_dec_rad = np.arcsin(np.clip(v_zenith_cirs[2], -1.0, 1.0))

        era_rad = cls.earth_rotation_angle(jd_utc, dut1)
        
        lat_rad = zenith_dec_rad
        lon_rad = (zenith_ra_rad - era_rad + np.pi) % (2 * np.pi) - np.pi

        return {
            "lat_deg": np.degrees(lat_rad),
            "lon_deg": np.degrees(lon_rad),
            "success": True
        }

    @staticmethod
    def decimal_to_dms(value):
        #decimal degrees to DMS
        degrees = int(abs(value))
        minutes_float = (abs(value) - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        return degrees, minutes, seconds
    
    @staticmethod
    def convert_coordinates(latitude, longitude, dms=False):
        #decimal lat/lon to N/S/E/W format
        lat_dir = "N" if latitude >= 0 else "S"
        lon_dir = "E" if longitude >= 0 else "W"
    
        if dms:
            lat_d, lat_m, lat_s = estimator.decimal_to_dms(latitude)
            lon_d, lon_m, lon_s = estimator.decimal_to_dms(longitude)
    
            return (
                f"{lat_d}° {lat_m}′ {lat_s:.2f}″ {lat_dir}, "
                f"{lon_d}° {lon_m}′ {lon_s:.2f}″ {lon_dir}"
            )
        else:
            return f"{abs(latitude):.6f}° {lat_dir}, {abs(longitude):.6f}° {lon_dir}"
        
    @staticmethod
    def centroid_matching_error_from_tetra3(sol):
        if "residuals_arcsec" not in sol:
            raise KeyError("tetra3 solution does not contain residuals arcseconds")
    
        residuals_arcsec = np.asarray(sol["residuals_arcsec"])
    
        rms_arcsec = sol.get("RMSE", np.sqrt(np.mean(residuals_arcsec**2)))
    
        return {
            "errors_arcsec": residuals_arcsec,
            "rms_arcsec": rms_arcsec,
            "n_matches": len(residuals_arcsec)
        }