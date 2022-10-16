#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/03/2019

@author: Maurizio Ferrari Dacrema
"""


def seconds_to_biggest_unit(time_in_seconds):

    conversion_factor_list = [
        ("sec", 1),
        ("min", 60),
        ("hour", 60),
        ("day", 24),
        ("year", 365),
    ]

    unit_index = 0
    temp_time_value = time_in_seconds
    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while temp_time_value >= 1.0 and unit_index < len(conversion_factor_list)-1:

        temp_time_value = temp_time_value/conversion_factor_list[unit_index+1][1]

        if temp_time_value >= 1.0:
            unit_index += 1
            new_time_value = temp_time_value
            new_time_unit = conversion_factor_list[unit_index][0]

    else:
        return new_time_value, new_time_unit

