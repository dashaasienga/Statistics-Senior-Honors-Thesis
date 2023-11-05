ppv <- function(p, sens, spec){
  ppv <- (sens*p)/((sens*p) + ((1-spec)*(1-p)))
  return(ppv)
}

npv <- function(p, sens, spec){
  npv <- (spec*(1-p))/(((1-sens)*p) + (spec*(1-p)))
  return(npv)
}

dat_8080 <- data.frame(prevalence = seq(0.05,0.95,0.05)
                       , sens=0.80
                       , spec=0.80
                       , ppv = ppv(p=seq(0.05,0.95,0.05), sens=0.80, spec=0.80)
                       , npv = npv(p=seq(0.05,0.95,0.05), sens=0.80, spec=0.80))

dat_9090 <- data.frame(prevalence = seq(0.05,0.95,0.05)
                       , sens=0.90
                       , spec=0.90
                       , ppv = ppv(p=seq(0.05,0.95,0.05), sens=0.90, spec=0.90)
                       , npv = npv(p=seq(0.05,0.95,0.05), sens=0.90, spec=0.90))

dat_9070 <- data.frame(prevalence = seq(0.05,0.95,0.05)
                       , sens=0.90
                       , spec=0.70
                       , ppv = ppv(p=seq(0.05,0.95,0.05), sens=0.90, spec=0.70)
                       , npv = npv(p=seq(0.05,0.95,0.05), sens=0.90, spec=0.70))

dat_7090 <- data.frame(prevalence = seq(0.05,0.95,0.05)
                       , sens=0.70
                       , spec=0.90
                       , ppv = ppv(p=seq(0.05,0.95,0.05), sens=0.70, spec=0.90)
                       , npv = npv(p=seq(0.05,0.95,0.05), sens=0.70, spec=0.90))

dat_all <- bind_rows(dat_8080, dat_7090, dat_9070, dat_9090) |>
  mutate(sens_spec = paste0("Sensitivity: ", sens, "\n Specificity: ", spec)
         , fpr = 1 - spec
         , fnr = 1 - sens)

ggplot(dat_all, aes(x=prevalence, y=ppv)) +
  geom_point() + 
  labs(x="prevalence", y="positive predictive value") +
  facet_wrap(~sens_spec)

ggplot(dat_all, aes(x=prevalence, y=npv)) +
  geom_point() + 
  labs(x="prevalence", y="negative predictive value") +
  facet_wrap(~sens_spec)


fpr <- function(p, ppv, sens){
  fpr <- (p/(1-p))*((1-ppv)/ppv)*sens
  return(fpr)
}

fpr_8080 <- data.frame(prevalence = seq(0.05,0.80,0.05)
                       , sens=0.80
                       , ppv=0.80
                       , fpr = fpr(p=seq(0.05,0.80,0.05), ppv=0.80, sens=0.80))

fpr_9090 <- data.frame(prevalence = seq(0.05,0.90,0.05)
                       , sens=0.90
                       , ppv=0.90
                       , fpr = fpr(p=seq(0.05,0.90,0.05), ppv=0.90, sens=0.90)) 

fpr_9070 <- data.frame(prevalence = seq(0.05,0.70,0.05)
                       , sens=0.90
                       , ppv=0.70
                       , fpr = fpr(p=seq(0.05,0.70,0.05), ppv=0.70, sens=0.90)) 

fpr_7090 <- data.frame(prevalence = seq(0.05,0.90,0.05)
                       , sens=0.70
                       , ppv=0.90
                       , fpr = fpr(p=seq(0.05,0.90,0.05), ppv=0.90, sens=0.70)) 

fpr_all <- bind_rows(fpr_8080, fpr_7090, fpr_9070, fpr_9090) |>
  mutate(ppv_sens = paste0("PPV: ", ppv, "\n Sensitivity: ", sens))

ggplot(fpr_all, aes(x=prevalence, y=fpr)) +
  geom_point() + 
  labs(x="prevalence", y="FPR") +
  facet_wrap(~ppv_sens)
