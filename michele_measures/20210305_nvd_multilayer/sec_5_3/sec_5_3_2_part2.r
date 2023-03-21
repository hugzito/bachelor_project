for (net in c("euair", "egosm", "copenhagen", "aarhus", "physics", "ira")) {
   print(net);
   mydata <- read.csv(paste("threshold_", net, ".csv", sep = ""), sep = "\t");

   m1 <- glm(beta ~ mlcount, data = mydata, family = poisson());

   m2 <- glm(beta ~ mlcount + mlge, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- glm(beta ~ mlcount + mlemd, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- glm(beta ~ mlcount + mlgft, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m1 <- glm(beta ~ count, data = mydata, family = poisson());

   m2 <- glm(beta ~ count + ge, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- glm(beta ~ count + emd, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- glm(beta ~ count + gft, data = mydata, family = poisson());
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));
}
