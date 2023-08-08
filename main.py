import os.path

import torch
import argparse

def showModels(dir: str, modelEnding="pt"):
    print("Searching...")
    data = os.walk(dir)
    counter = 0
    for dir in data:
        print("Dir: " + dir[0])
        print("--------------------")
        for model in dir[2]:
            if model.split(".")[-1] == modelEnding:
                print(model)
                counter += 1
        print("--------------------")
        print("Models found: " +  counter)
        print("--------------------")

def trainModel(loadModelDir: str, saveModelDir: str, device: str):
    print("TODO")

def runModel(loadModelDir: str, saveOutputAt: str, showOutput: bool, device: str):
    print("TODO")

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="AUTO", help="choose device to run models on. Available options: GPU, CPU and AUTO. AUTO preferes GPU but uses CPU if no GPU is found.")
    parser.add_argument("--selectAction", type=str, default="", help="Preselect option from menu")

    # arguments for training
    parser.add_argument("--trainLoad", type=str, default="", help="If selected Action = training (2), you can selelect an already trained Model to train further.")
    parser.add_argument("--trainSave", type=str, default="./models", help="Select dir to save model once trained.")

    # arguments for testing/runing a trained model
    parser.add_argument("--load", type=str, help="If selected Action = run Model (1), select Model to load.")
    parser.add_argument("--saveOutput", type=str, default="./output", help="If selected Action = run Model (1), select dir to save the output")
    parser.add_argument("--showOutput", type=bool, default=False, help="Open/Plot output of Model")

    # arguments for show models
    parser.add_argument("--showModelsDir", type=str, default=".", help="Specify dir to search for models.")

    args = parser.parse_args()


    # select CUDA device
    if torch.cuda.is_available() and (args.device.upper() == "GPU" or args.device.upper() == "AUTO"):
        device = torch.device('cuda')
        print("GPU active")
    else:
        device = torch.device('cpu')
        if args.device.upper() == "GPU": print("no GPU found continuing with CPU")
        else: print("CPU active")


    #Main Menu
    userInput = " "
    inputManualy = False
    while( not userInput == 4 ):

        if not args.selectAction.isnumeric() or int(args.selectAction) > 4 or int(args.selectAction) < 0:
            inputManualy = True
            print("--------------------")
            print("1: run Model")
            print("2: train Model")
            print("3: show saved Models")
            print("type '4' to close")
            valid = False
            while not valid:
                print("->", end="")
                userInput = input()
                if not userInput.isnumeric():
                    print("Invalid input")
                else:
                    valid = True
            userInput = int(userInput)
        else:
            if not args.selectAction.isnumeric():
                raise ValueError("'--selectAction' must be an int value")
            userInput = int(args.selectAction)


        if userInput == 1:
            if inputManualy:
                valid = False
                while not valid:
                    print("load Model: ", end="")
                    modelPath = input()
                    valid = os.path.exists(modelPath)
                    if not valid: print("Model not found!")
                valid = False
                while not valid:
                    print("save Output at: ", end="")
                    saveDir = input()
                    valid = os.path.exists(saveDir)
                    if not valid: print("Model not found!")
                valid = False
                print("show Output? true/FALSE: ", end="")
                showOutput = input()
                if showOutput.upper() == "TRUE":
                    showOutput = True
                else:
                    showOutput = False
                runModel(loadModelDir=modelPath, saveOutputAt=saveDir, showOutput=showOutput, device=device)
            else:
                if not args.load or not os.path.exists(args.load):
                    raise ValueError("'--load' is undefined or not found")
                if not os.path.exists(args.showOutput):
                    raise ValueError("'--showOutput' is undefined or not found")
                runModel(loadModelDir=args.load, saveOutputAt=args.saveOutput, showOutput=args.showOutput, device=device)
        elif userInput == 2:
            print("TRAIN MODEL")
            if inputManualy:
                valid = False
                while not valid:
                    print("load Model (emty for new Model): ", end="")
                    modelDir = input()
                    valid = os.path.exists(modelDir) #TODO check if file is valid model
                    if modelDir == "": valid = True
                    if not valid: print("Model not found!")

                valid = False
                while not valid:
                    print("save trained Model at: ", end="")
                    saveDir = input()
                    valid = os.path.exists(saveDir)
                    if not valid: print("dir not found")
                trainModel(loadModelDir=modelDir, saveModelDir=saveDir, device=device)
            else:
                if not args.trainSave or not os.path.exists(args.trainSave):
                    raise ValueError("--trainSave is not defined or not found")
                trainModel(loadModelDir=args.trainLoad, saveModelDir=args.trainSave, device=device)
        elif userInput == 3:
            if inputManualy:
                valid = False
                while not valid:
                    print("Enter dir to search (empty for all): ")
                    dirInput = input()
                    if dirInput == "":
                        dirInput = "."
                        valid = True
                    elif os.path.exists(dirInput):
                        valid = True
                showModels(dirInput)
            else:
                if not os.path.exists(args.showModelsDir):
                    raise ValueError("The directory specified in '--showModelsDir' does not exist")
                showModels(args.showModelsDir)

        elif userInput == 4:
            break

        else:
            print("invalid input, please try again")

        if not inputManualy:
            break
