for (net in c("euair", "egosm", "copenhagen", "aarhus", "physics", "ira")) {
   print(net);
   mydata <- read.csv(paste("cascade_", net, ".csv", sep = ""), sep = "\t");

   m1 <- lm(beta ~ mlcount, data = mydata);

   m2 <- lm(beta ~ mlcount + log(mlge), data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- lm(beta ~ mlcount + log(mlemd), data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- lm(beta ~ mlcount + log(mlgft), data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m1 <- lm(beta ~ count, data = mydata);

   m2 <- lm(beta ~ count + ge, data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- lm(beta ~ count + emd, data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));

   m2 <- lm(beta ~ count + gft, data = mydata);
   yerr1 <- mydata$beta - predict(m1, type = "response");
   yerr2 <- mydata$beta - predict(m2, type = "response");
   print((mean(abs(yerr1)) - mean(abs(yerr2))) / mean(abs(yerr1)));
}
