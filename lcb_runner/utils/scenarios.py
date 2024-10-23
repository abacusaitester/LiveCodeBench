from enum import Enum


class Scenario(Enum):
    codegeneration = "codegeneration"
    selfrepair = "selfrepair"
    testoutputprediction = "testoutputprediction"
    codeprofile = "codeprofile"
