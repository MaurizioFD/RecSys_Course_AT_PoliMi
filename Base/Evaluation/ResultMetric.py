#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/08/18

@author: Maurizio Ferrari Dacrema
"""


class ResultMetric(dict):
    """
    This class is a subclass of dict and implements a dictionary redefining its string in such a way to limit the number of decimals
    It can be used as if it were a normal dictionary
    """

    def __init__(self,*arg,**kw):

       super(ResultMetric, self).__init__(*arg, **kw)



    def __get_result_string(self):

        output_str = ""

        for metric in self.keys():
            output_str += "{}: {:.7f}, ".format(metric, self[metric])

        return output_str


    def __repr__(self):

        print("Calling redefined STR")

        return self.__get_result_string()



