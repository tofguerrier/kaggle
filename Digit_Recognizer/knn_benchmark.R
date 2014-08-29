# makes the KNN submission
library(FNN)

#Function to draw the image from csv
draw_digit <- function(set, id){
  size_img = 28
  #id = row
  #matrix containing the values to be plotted 
  mz = matrix(unlist(set[id,]),nrow = 28,ncol=28)
  image(x=1:size_img,y=1:size_img,z=mz,
        xlab = paste(" : ", id), ylab = "what")
}

#Read the data in
train <- read.csv("./data/train.csv", header=TRUE)
test <- read.csv("./data/test.csv", header=TRUE)

#Store label
labels <- train[,1]
train <- train[,-1]

#Take second element in train set which is labelled as a 3
#And plot it
draw_digit(train, which(labels == 3)[2])

#Run Knn as it
knn.results <- knn(train, test, labels, k = 10, algorithm="cover_tree")
results <- (0:9)[knn.results]

#Reformat for export submission
#write(results, file="knn_benchmark.csv", ncolumns=1) 
#Missing header and id for submissions
res = data.frame(ImageId = 1:28000, Label=results)


write.csv(res, file="knn_benchmark.csv", row.names=FALSE) 

#Submission result : Kaggle
#267  new	me	0.96557  Fri, 29 Aug 2014 11:45:13
