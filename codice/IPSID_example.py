### Numerical example for IPSID

# 1 - IPSID example with the main steps
# This is an example model (shown in Fig. 2A) with 
# (a) 2 behaviorally relevant latent states (behaviorally relevant intrinsic dynamics)
# (b) 4 other latent states (other intrinsic dynamics)
# (c) 2 states that drive the external input (input dynamics)

# 2 - IPSID example with the additional steps
# This is an example model (shown in Fig. 3) with
# (a) 2 behaviorally relevant latent states (behaviorally relevant intrinsic dynamics)
# (b) 2 other latent states (other intrinsic dynamics)
# (c) 2 states that drive the external input (input dynamics)
# (d) 2 behaviorally relevant latent states driven by the input u_k but not encoded in neural activity y_k


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import PSID
from PSID.evaluation import evalPrediction
from PSID.MatHelper import loadmat
from PSID.PrepModel import PrepModel


def main():

    # 1 - IPSID example with the main steps
    # Choose sample model
    sample_model_path = "sample_model_IPSID.mat"

    # Load data
    print("Loading example model from {}".format(sample_model_path))
    data = loadmat(sample_model_path)

    # Generate sample data from this model
    np.random.seed(42)  # For exact reproducibility

    N = int(2e5)

    # Generate dynamical input u
    uSys = PSID.LSSM(params=data["uSys"])
    u, _ = uSys.generateRealization(N)
    Au = uSys.A
    trueSys = PSID.LSSM(params=data["trueSys"])
    trueSys.Dz = trueSys.dz
    y, x = trueSys.generateRealization(N, u=u)
    z = (trueSys.Cz @ x.T).T + (trueSys.Dz @ u.T).T

    # Add some z dynamics that are not encoded in y (i.e. epsilon)
    epsSys = PSID.LSSM(params=data["epsSys"])
    _, x_eps = epsSys.generateRealization(N)
    z += (epsSys.Cz @ x_eps.T).T

    allYData, allZData, allUData = y, z, u

    # Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean.
    # Starting version v1.1.0, IPSID by default internally removes the mean from the neural/behavior data and adds
    # it back to predictions, so the user does not need to handle this preprocessing. If the data is already zero-mean,
    # this mean-removal will simply subtract and add zeros to signals so everything will still work.
    # To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data:
    YMean = 10 * np.random.randn(allYData.shape[-1])
    ZMean = 10 * np.random.randn(allZData.shape[-1])
    UMean = 10 * np.random.randn(allUData.shape[-1])
    allYData += YMean
    allZData += ZMean
    allUData += UMean
    # Also reflect this in the true model:
    trueSys.YPrepModel = PrepModel(mean=YMean, remove_mean=True)
    trueSys.ZPrepModel = PrepModel(mean=ZMean, remove_mean=True)
    trueSys.UPrepModel = PrepModel(mean=UMean, remove_mean=True)

    # Split data into training and test data:
    trainInds = np.arange(np.round(0.5 * allYData.shape[0]), dtype=int)
    testInds = np.arange(1 + trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]
    uTrain = allUData[trainInds, :]
    uTest = allUData[testInds, :]

    ## Example 1:
    # Extract only the intrinsic behaviorally relevant latent states (with nx = n1 = 2)
    idSys1 = PSID.IPSID(yTrain, zTrain, uTrain, nx=2, n1=2, i=10)

    # Predict behavior using the learned model
    zTestPred1, yTestPred1, xTestPred1 = idSys1.predict(yTest, uTest)

    # Compute R2 of decoding
    R2 = evalPrediction(zTest, zTestPred1, "R2")

    # Predict behavior using the true model for comparison
    zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = trueSys.predict(yTest, uTest)
    R2Ideal = evalPrediction(zTest, zTestPredIdeal, "R2")

    print("Behavior decoding R2:\n  IPSID => {:.3g}".format(np.mean(R2)))
    print("Ideal behavior decoding R2:\n  IPSID => {:.3g}".format(np.mean(R2Ideal)))

    ## Example 2:
    # Extract both behaviorally relevant and behaviorally irrelevant latent states (with nx = 6, n1 = 2)
    idSys2 = PSID.IPSID(yTrain, zTrain, uTrain, nx=6, n1=2, i=10)

    # This model will also have ideal neural self-prediction
    zTestPred2, yTestPred2, xTestPred2 = idSys2.predict(yTest, uTest)
    yR2 = evalPrediction(yTest, yTestPred2, "R2")
    yR2Ideal = evalPrediction(yTest, yTestPredIdeal, "R2")

    print("Neural self-prediction R2:\n  IPSID => {:.3g}".format(np.mean(yR2)))
    print("Ideal neural self-prediction R2:\n  IPSID => {:.3g}".format(np.mean(yR2Ideal)))

    ## Example 3:
    # IPSID can be used if data is available in discontinuous segments (e.g. different trials)
    # In this case, y, z and u data segments must be provided as elements of a list
    # Trials do not need to have the same number of samples
    # Here, for example assume that trials start at every 1000 samples.
    # And each each trial has a random length of 900 to 990 samples
    trialStartInds = np.arange(0, allYData.shape[0] - 1000, 1000)
    trialDurRange = np.array([900, 990])
    trialDur = np.random.randint(
        low=trialDurRange[0], high=1 + trialDurRange[1], size=trialStartInds.shape
    )
    trialInds = [
        trialStartInds[ti] + np.arange(trialDur[ti])
        for ti in range(trialStartInds.size)
    ]
    yTrials = [allYData[trialIndsThis, :] for trialIndsThis in trialInds]
    zTrials = [allZData[trialIndsThis, :] for trialIndsThis in trialInds]
    uTrials = [allUData[trialIndsThis, :] for trialIndsThis in trialInds]

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5 * len(yTrials)), dtype=int)
    testInds = np.arange(1 + trainInds[-1], len(yTrials))
    yTrainTrials = [yTrials[ti] for ti in trainInds]
    yTestTrials = [yTrials[ti] for ti in testInds]
    zTrainTrials = [zTrials[ti] for ti in trainInds]
    zTestTrials = [zTrials[ti] for ti in testInds]
    uTrainTrials = [uTrials[ti] for ti in trainInds]
    uTestTrials = [uTrials[ti] for ti in testInds]

    idSys3 = PSID.IPSID(yTrainTrials, zTrainTrials, uTrainTrials, nx=2, n1=2, i=10)

    zPredTrials, yPredTrials, xPredTrials = idSys3.predict(yTestTrials, uTestTrials)
    zPredTrialsIdeal, yPredTrialsIdeal, xPredTrialsIdeal = trueSys.predict(
        yTestTrials, uTestTrials
    )
    zTestA = np.concatenate(zTestTrials, axis=0)
    zPredA = np.concatenate(zPredTrials, axis=0)
    zPredIdealA = np.concatenate(zPredTrialsIdeal, axis=0)

    R2TrialBased = evalPrediction(zTestA, zPredA, "R2")
    R2TrialBasedIdeal = evalPrediction(zTestA, zPredIdealA, "R2")

    print("Behavior decoding R2 (trial-based learning/decoding):\n  IPSID => {:.3g}".format(np.mean(R2TrialBased)))
    print("Ideal behavior decoding R2 (trial-based learning/decoding):\n  IPSID => {:.3g}".format(np.mean(R2TrialBasedIdeal)))


    # #########################################
    # Plot the true and identified eigenvalues

    # Example 1:
    # Eigenvalues when only learning behaviorally relevant states
    idEigs1 = np.linalg.eig(idSys1.A)[0]

    # Example 2: 
    # Additional eigenvalues when also learning behaviorally irrelevant states
    idEigs2 = np.linalg.eig(idSys2.A[2:, 2:])[0]

    relevantDims = (
        trueSys.zDims - 1
    )  # Dimensions that drive both behavior and neural activity
    irrelevantDims = [
        x for x in np.arange(trueSys.state_dim, dtype=int) if x not in relevantDims
    ]  # Dimensions that only drive the neural activity
    trueEigsRelevant = np.linalg.eig(trueSys.A[np.ix_(relevantDims, relevantDims)])[0]
    trueEigsIrrelevant = np.linalg.eig(
        trueSys.A[np.ix_(irrelevantDims, irrelevantDims)]
    )[0]
    trueEigsInput = np.linalg.eig(Au)[0]

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove()
    ax = axs[0]
    ax.axis("equal")
    ax.add_patch(
        patches.Circle((0, 0), radius=1, fill=False, color="black", alpha=0.2, ls="-")
    )
    ax.plot([-1, 1, 0, 0, 0], [0, 0, 0, -1, 1], color="black", alpha=0.2, ls="-")
    ax.scatter(
        np.real(trueEigsInput),
        np.imag(trueEigsInput),
        marker="o",
        edgecolors="#800080",
        facecolors="none",
        label="Input eigenvalues",
    )
    ax.scatter(
        np.real(trueEigsIrrelevant),
        np.imag(trueEigsIrrelevant),
        marker="o",
        edgecolors="#FF5733",
        facecolors="none",
        label="Other neural eigenvalues",
    )
    ax.scatter(
        np.real(trueEigsRelevant),
        np.imag(trueEigsRelevant),
        marker="o",
        edgecolors="#50C878",
        facecolors="none",
        label="Behaviorally relevant neural eigenvalues",
    )
    ax.scatter(
        np.real(idEigs1),
        np.imag(idEigs1),
        marker="x",
        facecolors="#138a33",
        label="IPSID Identified (stage 1)",
    )
    ax.scatter(
        np.real(idEigs2),
        np.imag(idEigs2),
        marker="x",
        facecolors="#b04c1a",
        label="IPSID Identified (stage 2) - optional",
    )
    ax.set_title("True and identified eigevalues")
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()


    # 2 - IPSID example with the additional steps
    # Choose sample model
    sample_model_path = "sample_model_IPSID_add_step.mat"

    # Load data
    print("Loading example model from {}".format(sample_model_path))
    data = loadmat(sample_model_path)

    # Generate sample data from this model
    np.random.seed(42)  # For exact reproducibility

    N = int(2e5)

    # Generate dynamical input u
    uSys = PSID.LSSM(params=data["uSys"])
    u, _ = uSys.generateRealization(N)
    Au = uSys.A
    trueSys = PSID.LSSM(params=data["trueSys"])
    y, x = trueSys.generateRealization(N, u=u)
    z = (trueSys.Cz @ x.T).T

    allYData, allZData, allUData = y, z, u

    # Given the above state-space model used by IPSID, it is important for the neural/behavior/input data to be zero-mean.
    # Starting version v1.1.0, IPSID by default internally removes the mean from the neural/behavior data and adds
    # it back to predictions, so the user does not need to handle this preprocessing. If the data is already zero-mean,
    # this mean-removal will simply subtract and add zeros to signals so everything will still work.
    # To cover this general case with data that is not zero-mean, let's artificially add some non-zero mean to the sample data:
    YMean = 10 * np.random.randn(allYData.shape[-1])
    ZMean = 10 * np.random.randn(allZData.shape[-1])
    UMean = 10 * np.random.randn(allUData.shape[-1])
    allYData += YMean
    allZData += ZMean
    allUData += UMean
    # Also reflect this in the true model:
    trueSys.YPrepModel = PrepModel(mean=YMean, remove_mean=True)
    trueSys.ZPrepModel = PrepModel(mean=ZMean, remove_mean=True)
    trueSys.UPrepModel = PrepModel(mean=UMean, remove_mean=True)

    # Split data into training and test data:
    trainInds = np.arange(np.round(0.5 * allYData.shape[0]), dtype=int)
    testInds = np.arange(1 + trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]
    uTrain = allUData[trainInds, :]
    uTest = allUData[testInds, :]

    pass



if __name__ == "__main__":
    main()