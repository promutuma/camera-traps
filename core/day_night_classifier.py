"""
Day/Night Classification Module

Uses capture timestamp (OCR or file mtime) as the primary signal, with optional
sunrise/sunset refinement when station GPS is available. Falls back to pixel
brightness and night-vision detection when no timestamp is available.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from typing import Optional, Tuple

import cv2
import numpy as np


_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y",
]
_TIME_FORMATS = ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"]


def _parse_capture_datetime(
    capture_date: Optional[str],
    capture_time: Optional[str],
) -> Optional[datetime]:
    """Combine OCR date/time strings into a datetime."""
    if not capture_time:
        return None
    time_str = str(capture_time).strip().replace(".", ":")
    date_str = str(capture_date).strip() if capture_date else None

    if date_str:
        for dfmt in _DATE_FORMATS:
            for tfmt in _TIME_FORMATS:
                try:
                    return datetime.strptime(f"{date_str} {time_str}", f"{dfmt} {tfmt}")
                except ValueError:
                    continue

    for tfmt in _TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(time_str, tfmt).time()
            return datetime.combine(date.today(), parsed_time)
        except ValueError:
            continue

    match = re.match(r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?", time_str)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
            return datetime.combine(date.today(), time(hour, minute, second))
    return None


def _decimal_hour(dt: datetime) -> float:
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _sunrise_sunset_hours(lat_deg: float, lon_deg: float, on_date: date) -> Tuple[Optional[float], Optional[float]]:
    """
    Approximate sunrise/sunset as decimal hours in local solar time (NOAA-style).
    Returns (sunrise_hour, sunset_hour) or (None, None) on failure.
    """
    try:
        lat = math.radians(lat_deg)
        lon = lon_deg

        day_of_year = on_date.timetuple().tm_yday
        zenith = math.radians(90.833)  # official sun edge

        lng_hour = lon / 15.0
        rise_time = day_of_year + ((6 - lng_hour) / 24.0)
        set_time = day_of_year + ((18 - lng_hour) / 24.0)

        def _hour_angle(solar_time: float) -> Optional[float]:
            t = solar_time
            m = (0.9856 * t) - 3.289
            l = m + (1.916 * math.sin(math.radians(m))) + (0.020 * math.sin(math.radians(2 * m))) + 282.634
            l = l % 360.0
            ra = math.degrees(
                math.atan(0.91764 * math.tan(math.radians(l)))
            )
            ra = (ra + 360.0) % 360.0
            l_quadrant = math.floor(l / 90.0) * 90.0
            ra_quadrant = math.floor(ra / 90.0) * 90.0
            ra = ra + (l_quadrant - ra_quadrant)
            ra /= 15.0

            sin_dec = 0.39782 * math.sin(math.radians(l))
            cos_dec = math.cos(math.asin(sin_dec))

            cos_h = (math.cos(zenith) - (sin_dec * math.sin(lat))) / (cos_dec * math.cos(lat))
            if cos_h > 1 or cos_h < -1:
                return None
            h = math.degrees(math.acos(cos_h))
            if solar_time == rise_time:
                return (360.0 - h) / 15.0
            return h / 15.0

        rise = _hour_angle(rise_time)
        sunset = _hour_angle(set_time)
        if rise is None or sunset is None:
            return None, None

        # Convert from UTC to local solar time using longitude
        tz_offset = lon / 15.0
        sunrise_local = (rise - lng_hour + tz_offset) % 24.0
        sunset_local = (sunset - lng_hour + tz_offset) % 24.0
        return sunrise_local, sunset_local
    except Exception:
        return None, None


class DayNightClassifier:
    """Classifies images as day or night using timestamp-first logic."""

    def __init__(
        self,
        brightness_threshold: int = 100,
        day_start_hour: float = 6.0,
        day_end_hour: float = 18.0,
    ):
        self.brightness_threshold = brightness_threshold
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour

    def calculate_brightness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.mean(gray))

    def detect_night_vision(self, image: np.ndarray) -> bool:
        if len(image.shape) == 2:
            return True
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1])) < 30

    def classify_from_timestamp(
        self,
        capture_date: Optional[str],
        capture_time: Optional[str],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[str]:
        """Classify using OCR/file timestamp; returns None if time unavailable."""
        when = _parse_capture_datetime(capture_date, capture_time)
        if when is None:
            return None

        hour = _decimal_hour(when)

        if (
            latitude is not None
            and longitude is not None
            and capture_date
        ):
            sunrise, sunset = _sunrise_sunset_hours(latitude, longitude, when.date())
            if sunrise is not None and sunset is not None:
                if sunrise <= sunset:
                    in_daylight = sunrise <= hour <= sunset
                else:
                    # Polar edge case: sun doesn't set (or doesn't rise)
                    in_daylight = hour >= sunrise or hour <= sunset
                return "Day" if in_daylight else "Night"

        return "Day" if self.day_start_hour <= hour < self.day_end_hour else "Night"

    def classify_from_pixels(self, image: np.ndarray) -> str:
        is_night_vision = self.detect_night_vision(image)
        brightness = self.calculate_brightness(image)
        if is_night_vision:
            return "Night"
        return "Day" if brightness >= self.brightness_threshold else "Night"

    def classify(
        self,
        image_path: str,
        capture_date: Optional[str] = None,
        capture_time: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Classify an image as day or night.

        Priority:
          1. Capture timestamp (OCR / file mtime), with solar refinement when GPS known
          2. Pixel brightness and night-vision heuristics
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            brightness = self.calculate_brightness(image)

            timestamp_label = self.classify_from_timestamp(
                capture_date, capture_time, latitude, longitude
            )
            if timestamp_label:
                return timestamp_label, brightness

            return self.classify_from_pixels(image), brightness
        except Exception as e:
            print(f"Error classifying image {image_path}: {str(e)}")
            return ("Unknown", 0.0)

    def classify_with_confidence(
        self,
        image_path: str,
        capture_date: Optional[str] = None,
        capture_time: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Tuple[str, float, float]:
        classification, brightness = self.classify(
            image_path,
            capture_date=capture_date,
            capture_time=capture_time,
            latitude=latitude,
            longitude=longitude,
        )
        distance_from_threshold = abs(brightness - self.brightness_threshold)
        max_distance = (
            255 - self.brightness_threshold
            if brightness >= self.brightness_threshold
            else self.brightness_threshold
        )
        confidence = (
            min(distance_from_threshold / max_distance, 1.0)
            if max_distance > 0
            else 0.5
        )
        return classification, brightness, confidence


def classify_day_night(
    image_path: str,
    brightness_threshold: int = 100,
    capture_date: Optional[str] = None,
    capture_time: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Tuple[str, float]:
    classifier = DayNightClassifier(brightness_threshold=brightness_threshold)
    return classifier.classify(
        image_path,
        capture_date=capture_date,
        capture_time=capture_time,
        latitude=latitude,
        longitude=longitude,
    )
