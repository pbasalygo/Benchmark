import Swift
import Foundation
import CreateML

let calendar = Calendar(identifier: .gregorian)
let dateFormatter = DateFormatter()
dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss:Z"
dateFormatter.timeZone = calendar.timeZone

let line = "--------------------------------------------------"


func createImageClassifier(data: MLImageClassifier.DataSource) -> (classifier: MLImageClassifier, time: Double) {
    let start = DispatchTime.now()
    let createdClassifier = try! MLImageClassifier(trainingData: data)
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (createdClassifier, timeInterval)
}

func testImageClassifier(data: MLImageClassifier.DataSource, classifier: MLImageClassifier) -> (accuracy: Double, time: Double){
    let start = DispatchTime.now()
    let evaluationMetrics = classifier.evaluation(on: data)
    let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (evaluationAccuracy, timeInterval)
}

func createImageModel(dataName: String) {
    let trainingDataPath = Bundle.main.path(forResource: "Training", ofType: nil, inDirectory: dataName)!
    let testingDataPath = Bundle.main.path(forResource: "Testing", ofType: nil, inDirectory: dataName)!
    
    let trainingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: trainingDataPath))
    let testingData = MLImageClassifier.DataSource.labeledDirectories(at: URL(fileURLWithPath: testingDataPath))
    
    let createdClassifier = createImageClassifier(data: trainingData)
    let trainingAccuracy = (1.0 - createdClassifier.classifier.trainingMetrics.classificationError) * 100
    let validationAccuracy = (1.0 - createdClassifier.classifier.validationMetrics.classificationError) * 100
    print(line)
    print("Czas tworzenia klasyfikatora: \(createdClassifier.time) [s]")
    print("Poprawność treningu: \(trainingAccuracy)%, poprawność walidacji: \(validationAccuracy)%")
    print(line)
    
    let classifierTests = testImageClassifier(data: testingData, classifier: createdClassifier.classifier)
    print(line)
    print("Czas testowania klasyfikatora: \(classifierTests.time) [s], poprawność: \(classifierTests.accuracy)%")
}

func createTabularClassifier(data: MLDataTable, targetColumn: String) -> (classifier: MLClassifier, time: Double) {
    let start = DispatchTime.now()
    let createdClassifier = try! MLClassifier(trainingData: data, targetColumn: targetColumn)
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (createdClassifier, timeInterval)
}

func testTabularClassifier(data:  MLDataTable, classifier: MLClassifier) -> (accuracy: Double, time: Double){
    let start = DispatchTime.now()
    let evaluationMetrics = classifier.evaluation(on: data)
    let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (evaluationAccuracy, timeInterval)
}

func createTabularClassifierModel() {
    let dataFile = Bundle.main.url(forResource: "PaymentFraud", withExtension: "csv")!
    let dataTable = try! MLDataTable(contentsOf: dataFile)
    
    let featureColumns = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "isFraud"]
    let cleanedDataTable = dataTable[featureColumns]
    
    let (testingData, trainingData) = cleanedDataTable.randomSplit(by: 0.20, seed: 5)
    print(line)
    let createdClassifier = createTabularClassifier(data: trainingData, targetColumn: "isFraud")
    let trainingAccuracy = (1.0 - createdClassifier.classifier.trainingMetrics.classificationError) * 100
    let validationAccuracy = (1.0 - createdClassifier.classifier.validationMetrics.classificationError) * 100
    print(line)
    print("Czas tworzenia klasyfikatora: \(createdClassifier.time) [s]")
    print("Poprawność treningu: \(trainingAccuracy)%, poprawność walidacji: \(validationAccuracy)%")
    print(line)
    let classifierTests = testTabularClassifier(data: testingData, classifier: createdClassifier.classifier)
    print("Czas testowania klasyfikatora: \(classifierTests.time) [s], poprawność: \(classifierTests.accuracy)%")
    
}

func createTabularRegressor(data: MLDataTable, targetColumn: String) -> (regressor: MLLinearRegressor, time: Double) {
    let start = DispatchTime.now()
    let createdRegressor = try! MLLinearRegressor(trainingData: data.dropMissing(), targetColumn: targetColumn)
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (createdRegressor, timeInterval)
}

func testTabularRegressor(data:  MLDataTable, regressor: MLLinearRegressor) -> (worstEvaluationError: Double, RMSE: Double, time: Double){
    let start = DispatchTime.now()
    let evaluationMetrics = regressor.evaluation(on: data)
    let worstEvaluationError = evaluationMetrics.maximumError
    let testingRMSE = evaluationMetrics.rootMeanSquaredError
    let end = DispatchTime.now()
    
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    return (worstEvaluationError, testingRMSE, timeInterval)
}

func createTabularRegressorModel() {
    let dataFile = Bundle.main.url(forResource: "SteamReviewsCleaned", withExtension: "csv")!
    let dataTable = try! MLDataTable(contentsOf: dataFile)
    
//    let featureColumns = ["Store", "IsHoliday", "Dept", "Weekly_Sales", "Temperature", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment", "Type", "Size"]
    let featureColumns = ["language", "recommended", "votes_helpful", "votes_funny", "weighted_vote_score", "comment_count", "steam_purchase", "received_for_free", "written_during_early_access", "author.num_games_owned", "author.num_reviews", "author.playtime_forever", "author.playtime_last_two_weeks", "author.playtime_at_review", "author.last_played"]

    let cleanedDataTable = dataTable[featureColumns]
    
    let (testingData, trainingData) = cleanedDataTable.randomSplit(by: 0.20, seed: 5)
    print(line)
    let createdRegressor = createTabularRegressor(data: trainingData, targetColumn: "author.playtime_forever")
    let worstTrainingError = createdRegressor.regressor.trainingMetrics.maximumError
    let worstValidationError = createdRegressor.regressor.validationMetrics.maximumError
    let trainingRMSE = createdRegressor.regressor.trainingMetrics.rootMeanSquaredError
    let validationRMSE = createdRegressor.regressor.validationMetrics.rootMeanSquaredError
    print(line)
    print("Czas tworzenia regresora: \(createdRegressor.time) [s]")
    print("Największy błąd treningu: \(worstTrainingError), największy błąd walidacji: \(worstValidationError)s")
    print("RMSE treningu: \(trainingRMSE), RMSE walidacji: \(validationRMSE)s")
    print(line)
    let regressorTests = testTabularRegressor(data: testingData, regressor: createdRegressor.regressor)
    print("Czas testowania regresora: \(regressorTests.time) [s], największy błąd tesotwania: \(regressorTests.worstEvaluationError)")
    print("RMSE testowania: \(regressorTests.RMSE)")
    
}


print("Benchmarker")
let startupTimestamp = dateFormatter.string(from:Date())
print("Data i czas pomiaru: \(startupTimestamp)")
let benchmark_start = DispatchTime.now()

for iteration_number in 1...3 {
    
    print(line)
    print("Rozpoczęto iterację numer: \(iteration_number)")
    print(line)
    
    let start = DispatchTime.now()

    print("Model: ClassifierData")
    createImageModel(dataName: "ClassifierData")

    print(line)
    print("Model: Animals")
    createImageModel(dataName: "Animals")

    print(line)
    print("Model: PaymentFraud")
    createTabularClassifierModel()

    print(line)
    print("Model: SteamReviews")
    createTabularRegressorModel()
    print(line)

    let end = DispatchTime.now()
    let timeInterval = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    print("Czas iteracji numer \(iteration_number): \(timeInterval) [s]")
}

let benchmark_end = DispatchTime.now()
let benchmarkTime = Double(benchmark_end.uptimeNanoseconds - benchmark_start.uptimeNanoseconds) / 1_000_000_000
print("Całkowity czas pomiarów: \(benchmarkTime) [s]")

let endingTimestamp = dateFormatter.string(from:Date())
print("Czas zakończenia: \(endingTimestamp)")
print("Benchmarker zakończył działanie.")
