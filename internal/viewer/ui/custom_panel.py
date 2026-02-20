import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re


class CustomPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        self._setup_point_cloud_folder()
        self._setup_gaussian_edit_folder()
        self._setup_save_gaussian_folder()