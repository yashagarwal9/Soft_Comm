#!/usr/bin/env python
import os
import sys
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size[0])
            self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
            self.fc3 = nn.Linear(hidden_size[1], num_classes)
            self.relu = nn.ReLU()
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return out
        
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "survey_form.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    execute_from_command_line(sys.argv)
