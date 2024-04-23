import pandas as pd

currentTimestamp = pd.Timestamp.now()
logFileName = "logs/" + currentTimestamp.strftime("%Y-%m-%d-%H-%M-%S") + ".log"