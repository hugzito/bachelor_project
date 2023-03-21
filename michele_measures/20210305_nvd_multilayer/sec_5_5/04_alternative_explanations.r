mydata <- read.csv("layer_centralization.csv", sep = "\t");

mymodel <- lm(dist ~ edge_count, data = mydata);
print(summary(mymodel));

mymodel <- lm(dist ~ degcentr, data = mydata);
print(summary(mymodel));

mymodel <- lm(dist ~ closecentr, data = mydata);
print(summary(mymodel));

mymodel <- lm(dist ~ betcentr, data = mydata);
print(summary(mymodel));

