
### --- Notes:
#
# This script will receive args from
#   wrapper script, and make timing/duration
#   files for vCAT loc and study (task) phases.
#
# New study timing files only models conditional,
#   fixed, and fixed-BL trials
#
# Update - added fixed-BL trials preceding face
#   and scene for SVM categories


func_locTime <- function(numRuns, phase, subjStr, dataDir, outDir){
  
  ### --- Notes:
  #
  # This will make timing and duration files for face, 
  #   scene, and number times during localizer (orienting)
  #   task.
  #
  # Output format is 1 row per run (AFNI style).
  #
  # Timing file is tf_phase_foo.txt
  #   phase = experiment phase, foo = behavior
  #
  # Duration file is dur_phase_foo.txt
  #   Only a single duration is written per run
  #   Since all timing is fixed (no duration modulation).
  
  for(run in 1:numRuns){
    
    # determine append
    if (run == 1) {
      h_ap <- F
    } else {
      h_ap <- T
    }
    
    # get data
    data_raw <- read.delim(
      paste0(dataDir,"/", subjStr, "_simp_", phase, run, ".csv"), 
      sep = ","
    )
    
    # get indices
    ind_face <- grep("face_img", data_raw$stim_img)
    ind_scene <- grep("scene_img", data_raw$stim_img)
    ind_num <- grep("^[0-9]", data_raw$stim_img)
    
    # determine block lengths
    for(stim in c("face", "scene", "num")){
      
      h_ind <- get(paste0("ind_",stim))
      
      h_block_end <- vector()
      h_block_start <- vector()
      for(i in 1:length(h_ind)){
        
        # get start
        if(i == 1){
          h_block_start <- c(h_block_start, h_ind[i])
        }
        
        # find break points
        if(h_ind[i]+1 <= range(h_ind)[2]){
          if(h_ind[i+1] > h_ind[i]+1){
            h_block_end <- c(h_block_end, h_ind[i])
            h_block_start <- c(h_block_start, h_ind[i+1])
          }
        }else{
          # get end
          h_block_end <- c(h_block_end, h_ind[i])
        }
      }
      
      # make row - marry onset, duration
      if(length(h_block_end) == length(h_block_start)){
        row_input <- vector()
        val_input <- vector()
        for(i in 1:length(h_block_start)){
          val_start <- round(as.numeric(data_raw$onset[h_block_start[i]]),1)
          val_dur <- round(
            as.numeric(data_raw$onset[h_block_end[i]]) - 
            as.numeric(data_raw$onset[h_block_start[i]]), 
          1)
          # row_input <- c(row_input, paste0(val_start,":",val_dur))
          row_input <- c(row_input, val_start)
          val_input <- c(val_input, val_dur)
        }
      }
      
      # update -- get most common time rather
      #   than making table of durations since all are
      #   appx equal
      h_dur <- sort(table(val_input), decreasing=T)[1]
      dur_input <- names(h_dur)
      
      # write
      out_file <- paste0(outDir, "/tf_loc_", stim,".txt")
      cat(row_input, "\n", file = out_file, append = h_ap, sep = "\t")
      out_dur <- paste0(outDir, "/dur_loc_", stim,".txt")
      cat(dur_input, "\n", file = out_dur, append = h_ap, sep = "\t")
    }
  }
}

func_studyTime_old <- function(numRuns, phase, subjStr, dataDir, outDir){
  
  # ENI duration
  end_dur <- 0.7
  
  for(run in 1:numRuns){
    
    # determine append
    if (run == 1) {
      h_ap <- F
    } else {
      h_ap <- T
    }
    
    # get data
    data_raw <- read.delim(paste0(dataDir, "/", subjStr, "_simp_", phase, run, ".csv"), sep=",")
    
    # replace NA with NR for BL types
    data_raw$k_img[is.na(data_raw$k_img)] <- "NR"
    data_raw$stim[is.na(data_raw$stim)] <- "NR"
    
    # determine onsets for type (BL, cond)
    #   face vs scene
    #   responded, correct vs incorrect
    #   event and preceding event
    for(type in c("cond", "BL")){
      
      # get indices
      ind_hold <- grep(type, data_raw$trialtype)
      
      # start vectors
      for(i in c("face", "scene")){
        for(j in c("prec", "event")){
          if(type == "cond"){
            for(k in c("cor", "icor")){
              
              assign(paste0("ons_",type,"_",i,"_",j,"_",k), vector())
              assign(paste0("dur_",type,"_",i,"_",j,"_",k), vector())
              
              if(j == "prec"){
                assign(paste0("ons_fixed_",i,"_",j,"_",k), vector())
                assign(paste0("dur_fixed_",i,"_",j,"_",k), vector())
              }
            }
          }else{
            assign(paste0("ons_",type,"_",i,"_",j), vector())
            assign(paste0("dur_",type,"_",i,"_",j), vector())
          }
        }
      }
      
      # fill vectors
      #   event = actual trial, precede = trial preceding event
      for(ind in ind_hold){
        
        # account for end of file
        #   if last trial is cond, hardcode duration (2.3s)
        if(is.na(data_raw$onset[ind+1])==F){
          h_ons_next <- round(data_raw$onset[ind+1], 1)
        }else{
          h_ons_next < round(data_raw$onset[ind] + 2.3, 1)
        }
        
        # calc onset:dur for event
        #   exclude feedback time (0.5s) and ISI (0.6s)
        hold_event_onset <- round(data_raw$onset[ind],1)
        hold_event_dur <- round(h_ons_next - hold_event_onset - end_dur, 1)
        # hold_event_marry <- paste0(hold_event_onset,":",hold_event_dur)
        
        # calc onset:dur for precede
        hold_prec_onset <- round(data_raw$onset[ind-1],1)
        hold_prec_dur <- round(h_ons_next - hold_prec_onset, 1)
        
        # calc onset:dur for fixed
        #   exclude feedback time
        hold_fixed_onset <- round(data_raw$onset[ind-1],1)
        hold_fixed_dur <- round(hold_event_onset - hold_fixed_onset - end_dur, 1)
        
        # control response
        if(data_raw$resp[ind-1] != "None"){
          
          # control for type
          #   cond has accuracy, BL does not
          if(type == "cond"){
            
            # control for stimulus
            if(data_raw$stim[ind-1] == "face1" || data_raw$stim[ind-1] == "face2"){
              
              # control for accuracy
              if(data_raw$acc[ind-1] == 1){
                
                # dynamically append event row onset
                hold_event <- get(paste0("ons_",type,"_face_event_cor"))
                # assign(paste0("ons_",type,"_face_event_cor"), c(hold_event, hold_event_marry))
                assign(paste0("ons_",type,"_face_event_cor"), c(hold_event, hold_event_onset))
                
                # dynamically append event row duration
                hold_edur <- get(paste0("dur_",type,"_face_event_cor"))
                assign(paste0("dur_",type,"_face_event_cor"), c(hold_edur, hold_event_dur))
                
                # dynamically append prec row onset
                hold_prec <- get(paste0("ons_",type,"_face_prec_cor"))
                assign(paste0("ons_",type,"_face_prec_cor"), c(hold_prec, hold_prec_onset))
                
                # dynamically append prec row duration
                hold_pdur <- get(paste0("dur_",type,"_face_prec_cor"))
                assign(paste0("dur_",type,"_face_prec_cor"), c(hold_pdur, hold_prec_dur))
                
                # append fixed onset, duration
                ons_fixed_face_prec_cor <- c(ons_fixed_face_prec_cor, hold_fixed_onset)
                dur_fixed_face_prec_cor <- c(dur_fixed_face_prec_cor, hold_fixed_dur)
                
              }else{
                
                # onset (prec, cond)
                hold_event <- get(paste0("ons_",type,"_face_event_icor"))
                assign(paste0("ons_",type,"_face_event_icor"), c(hold_event, hold_event_onset))
                hold_prec <- get(paste0("ons_",type,"_face_prec_icor"))
                assign(paste0("ons_",type,"_face_prec_icor"), c(hold_prec, hold_prec_onset))
                
                # duration (prec, cond)
                hold_edur <- get(paste0("dur_",type,"_face_event_icor"))
                assign(paste0("dur_",type,"_face_event_icor"), c(hold_edur, hold_event_dur))
                hold_pdur <- get(paste0("dur_",type,"_face_prec_icor"))
                assign(paste0("dur_",type,"_face_prec_icor"), c(hold_pdur, hold_prec_dur))
                
                # fixed onset, duration
                ons_fixed_face_prec_icor <- c(ons_fixed_face_prec_icor, hold_fixed_onset)
                dur_fixed_face_prec_icor <- c(dur_fixed_face_prec_icor, hold_fixed_dur)
              }
              
            }else if(data_raw$stim[ind-1] == "scene1" || data_raw$stim[ind-1] == "scene2"){
              if(data_raw$acc[ind-1] == 1){
                
                # onset
                hold_event <- get(paste0("ons_",type,"_scene_event_cor"))
                assign(paste0("ons_",type,"_scene_event_cor"), c(hold_event, hold_event_onset))
                hold_prec <- get(paste0("ons_",type,"_scene_prec_cor"))
                assign(paste0("ons_",type,"_scene_prec_cor"), c(hold_prec, hold_prec_onset))
                
                # duration
                hold_edur <- get(paste0("dur_",type,"_scene_event_cor"))
                assign(paste0("dur_",type,"_scene_event_cor"), c(hold_edur, hold_event_dur))
                hold_pdur <- get(paste0("dur_",type,"_scene_prec_cor"))
                assign(paste0("dur_",type,"_scene_prec_cor"), c(hold_pdur, hold_prec_dur))
                
                # fixed onset, duration
                ons_fixed_scene_prec_cor <- c(ons_fixed_scene_prec_cor, hold_fixed_onset)
                dur_fixed_scene_prec_cor <- c(dur_fixed_scene_prec_cor, hold_fixed_dur)
                
              }else{
                
                # onset
                hold_event <- get(paste0("ons_",type,"_scene_event_icor"))
                assign(paste0("ons_",type,"_scene_event_icor"), c(hold_event, hold_event_onset))
                hold_prec <- get(paste0("ons_",type,"_scene_prec_icor"))
                assign(paste0("ons_",type,"_scene_prec_icor"), c(hold_prec, hold_prec_onset))
                
                # duration
                hold_edur <- get(paste0("dur_",type,"_scene_event_icor"))
                assign(paste0("dur_",type,"_scene_event_icor"), c(hold_edur, hold_event_dur))
                hold_pdur <- get(paste0("dur_",type,"_scene_prec_icor"))
                assign(paste0("dur_",type,"_scene_prec_icor"), c(hold_pdur, hold_prec_dur))
                
                # fixed onset, duration
                ons_fixed_scene_prec_icor <- c(ons_fixed_scene_prec_icor, hold_fixed_onset)
                dur_fixed_scene_prec_icor <- c(dur_fixed_scene_prec_icor, hold_fixed_dur)
              }
            }
            
          }else{
            
            # BL
            if(data_raw$stim[ind-1] == "face1" || data_raw$stim[ind-1] == "face2"){
              
              # onset
              hold_event <- get(paste0("ons_",type,"_face_event"))
              assign(paste0("ons_",type,"_face_event"), c(hold_event, hold_event_onset))
              hold_prec <- get(paste0("ons_",type,"_face_prec"))
              assign(paste0("ons_",type,"_face_prec"), c(hold_prec, hold_prec_onset))
              
              # duration
              hold_edur <- get(paste0("dur_",type,"_face_event"))
              assign(paste0("dur_",type,"_face_event"), c(hold_edur, hold_event_dur))
              hold_pdur <- get(paste0("dur_",type,"_face_prec"))
              assign(paste0("dur_",type,"_face_prec"), c(hold_pdur, hold_prec_dur))
              
            }else if(data_raw$stim[ind-1] == "scene1" || data_raw$stim[ind-1] == "scene2"){
              
              # onset
              hold_event <- get(paste0("ons_",type,"_scene_event"))
              assign(paste0("ons_",type,"_scene_event"), c(hold_event, hold_event_onset))
              hold_prec <- get(paste0("ons_",type,"_scene_prec"))
              assign(paste0("ons_",type,"_scene_prec"), c(hold_prec, hold_prec_onset))
              
              # duration
              hold_edur <- get(paste0("dur_",type,"_scene_event"))
              assign(paste0("dur_",type,"_scene_event"), c(hold_edur, hold_event_dur))
              hold_pdur <- get(paste0("dur_",type,"_scene_prec"))
              assign(paste0("dur_",type,"_scene_prec"), c(hold_pdur, hold_prec_dur))
              
            } # close stimulus cond
          } # close type cond
        } # close resp cond
        
      } # close ind loop
      
      # write out
      for(i in c("face", "scene")){
        for(j in c("prec", "event")){
          if(type == "cond"){
            for(k in c("cor", "icor")){
              
              # get dynamic variable
              hold_out <- get(paste0("ons_",type,"_",i,"_",j,"_",k))
              hold_dur <- get(paste0("dur_",type,"_",i,"_",j,"_",k))
              
              # deal with NR
              if(length(hold_out) == 0){
                hold_out <- "*"
                hold_dur <- "*"
              }else{
                
                # update -- get most common time rather
                #   than making table of durations since all are
                #   appx equal
                h_dur <- sort(table(hold_dur), decreasing=T)[1]
                hold_dur <- names(h_dur)
              }
              
              # set out file, write
              out_file <- paste0(outDir, "/tf_Study_", substr(type,1,1), substr(i,1,1), substr(j,1,1), substr(k,1,1),".txt")
              cat(hold_out, "\n", file = out_file, append = h_ap, sep = "\t")
              out_dur <- paste0(outDir, "/dur_Study_", substr(type,1,1), substr(i,1,1), substr(j,1,1), substr(k,1,1),".txt")
              cat(hold_dur, "\n", file = out_dur, append = h_ap, sep = "\t")
              
              # repeat for fixed tf
              if(j == "prec"){
                
                hold_out <- get(paste0("ons_fixed_", i, "_", j, "_", k))
                hold_dur <- get(paste0("dur_fixed_", i, "_", j, "_", k))
                
                if(length(hold_out) == 0){
                  hold_out <- "*"
                  hold_dur <- "*"
                }else{
                  h_dur <- sort(table(hold_dur), decreasing=T)[1]
                  hold_dur <- names(h_dur)
                }
                
                h_dur <- sort(table(hold_dur), decreasing=T)[1]
                hold_dur <- names(h_dur)
                
                out_file <- paste0(outDir, "/tf_Study_F", substr(i,1,1), substr(j,1,1), substr(k,1,1),".txt")
                cat(hold_out, "\n", file = out_file, append = h_ap, sep = "\t")
                out_dur <- paste0(outDir, "/dur_Study_F", substr(i,1,1), substr(j,1,1), substr(k,1,1),".txt")
                cat(hold_dur, "\n", file = out_dur, append = h_ap, sep = "\t")
              }
            }
            
          }else{
            
            hold_out <- get(paste0("ons_",type,"_",i,"_",j))
            hold_dur <- get(paste0("dur_",type,"_",i,"_",j))
            
            if(length(hold_out) == 0){
              hold_out <- "*"
              hold_dur <- "*"
            }else{
              h_dur <- sort(table(hold_dur), decreasing=T)[1]
              hold_dur <- names(h_dur)
            }
            
            out_file <- paste0(outDir, "/tf_Study_", substr(type,1,1), substr(i,1,1), substr(j,1,1), ".txt")
            cat(hold_out, "\n", file = out_file, append = h_ap, sep = "\t")
            out_dur <- paste0(outDir, "/dur_Study_", substr(type,1,1), substr(i,1,1), substr(j,1,1), ".txt")
            cat(hold_dur, "\n", file = out_dur, append = h_ap, sep = "\t")
          }
        }
      }
    } # close type loop
  } # close run loop
}

func_studyTime <- function(numRuns, phase, subjStr, dataDir, outDir){
  
  ### --- Notes:
  #
  # This will make timing and duration files for fixed,
  #   conditional, and fixed trials preceding baseline trials
  #
  # Fixed will not include fixed-BL trials
  
  # isi duration
  isi_dur <- 0.8
  
  for(run in 1:numRuns){
    
    # determine append
    if (run == 1) {
      h_ap <- F
    } else {
      h_ap <- T
    }
    
    # get data
    data_raw <- read.delim(
      paste0(dataDir, "/", subjStr, "_simp_", phase, run, ".csv"), 
      sep=","
    )
    
    # replace NA with NR for BL types
    data_raw$k_img[is.na(data_raw$k_img)] <- "NR"
    data_raw$stim[is.na(data_raw$stim)] <- "NR"
    
    # determine location of stim types
    ind_nr <- which(data_raw$resp == "None")
    ind_con <- which(data_raw$trialtype == "cond" & data_raw$resp != "None")
    ind_fix <- which(data_raw$trialtype == "fixed" & data_raw$resp != "None")
    
    # determine whether fix precedes BL
    h_fix_bl <- which(data_raw[ind_fix + 1,]$trialtype == "BL")
    ind_fix_bl <- ind_fix[h_fix_bl]
    h_fix_nbl <- which(data_raw[ind_fix + 1,]$trialtype != "BL")
    ind_fix_nbl <- ind_fix[h_fix_nbl]
    
    # determine fbl scene vs face
    h_fix_bl_face <- which(
      data_raw[ind_fix_bl + 2,]$stim == "face1" |
      data_raw[ind_fix_bl + 2,]$stim == "face2"
    )
    ind_fix_bl_face <- ind_fix_bl[h_fix_bl_face]
    
    h_fix_bl_scene <- which(
      data_raw[ind_fix_bl + 2,]$stim == "scene1" |
      data_raw[ind_fix_bl + 2,]$stim == "scene2"
    )
    ind_fix_bl_scene <- ind_fix_bl[h_fix_bl_scene]
    
    # grab onset, calc duration, don't include ISI
    #   Note - fix_bl increases index by 2
    #   Also, most common duration is captured
    
    # con
    ons_con <- data_raw[ind_con,]$onset
    h_dur_con <- round(data_raw[ind_con + 1,]$onset - ons_con - isi_dur, 1)
    dur_con <- as.numeric(names(table(h_dur_con)[1]))
    
    # fbl
    ons_fix_bl <- data_raw[ind_fix_bl,]$onset
    h_dur_fix_bl <- round(
      data_raw[ind_fix_bl + 2,]$onset - ons_fix_bl - isi_dur, 
      1
    )
    dur_fix_bl <- as.numeric(names(table(h_dur_fix_bl)[1]))
    
    # fbl-face
    ons_fix_bl_face <- data_raw[ind_fix_bl_face,]$onset
    dur_fix_bl_face <- dur_fix_bl
    
    # fbl-scene
    ons_fix_bl_scene <- data_raw[ind_fix_bl_scene,]$onset
    dur_fix_bl_scene <- dur_fix_bl
    
    # fix (no bl)
    ons_fix_nbl <- data_raw[ind_fix_nbl,]$onset
    h_dur_fix_nbl <- round(
      data_raw[ind_fix_nbl + 1,]$onset - ons_fix_nbl - isi_dur, 
      1
    )
    dur_fix_nbl <- as.numeric(names(table(h_dur_fix_nbl)[1]))
    
    # write out timing files
    out_tf_con <- paste0(outDir, "/tf_Study_con.txt")
    out_tf_fix <- paste0(outDir, "/tf_Study_fix.txt")
    out_tf_fbl <- paste0(outDir, "/tf_Study_fbl.txt")
    out_tf_fblf <- paste0(outDir, "/tf_Study_fblf.txt")
    out_tf_fbls <- paste0(outDir, "/tf_Study_fbls.txt")
    
    cat(
      round(ons_con, 1), 
      "\n", 
      file = out_tf_con, 
      append = h_ap, 
      sep = "\t"
    )
    cat(
      round(ons_fix_nbl, 1), 
      "\n", 
      file = out_tf_fix, 
      append = h_ap, 
      sep = "\t"
    )
    cat(
      round(ons_fix_bl, 1), 
      "\n", 
      file = out_tf_fbl, 
      append = h_ap, 
      sep = "\t"
    )
    cat(
      round(ons_fix_bl_face, 1), 
      "\n", 
      file = out_tf_fblf, 
      append = h_ap, 
      sep = "\t"
    )
    cat(
      round(ons_fix_bl_scene, 1), 
      "\n", 
      file = out_tf_fbls, 
      append = h_ap, 
      sep = "\t"
    )
    
    # write out duration files
    out_dur_con <- paste0(outDir, "/dur_Study_con.txt")
    out_dur_fix <- paste0(outDir, "/dur_Study_fix.txt")
    out_dur_fbl <- paste0(outDir, "/dur_Study_fbl.txt")
    out_dur_fblf <- paste0(outDir, "/dur_Study_fblf.txt")
    out_dur_fbls <- paste0(outDir, "/dur_Study_fbls.txt")
    
    cat(dur_con, "\n", file = out_dur_con, append = h_ap, sep = "\t")
    cat(dur_fix_nbl, "\n", file = out_dur_fix, append = h_ap, sep = "\t")
    cat(dur_fix_bl, "\n", file = out_dur_fbl, append = h_ap, sep = "\t")
    cat(dur_fix_bl_face, "\n", file = out_dur_fblf, append = h_ap, sep = "\t")
    cat(dur_fix_bl_scene, "\n", file = out_dur_fbls, append = h_ap, sep = "\t")
  }
}


# Get args from wrapper
args <- commandArgs()
dataDir <- args[6]
outDir <- args[7]
subjStr <- args[8]
numRuns <- args[9]
phase <- args[10]

# # For testing
# subjStr <- "vCAT_008"
# dataDir <- paste0("/Users/nmuncy/Projects/learn_mvpa/vCAT_data/", subjStr)
# outDir <- "~/Desktop/vCAT_time"
# numRuns <- 2
# phase <- "task"
# run <- 1


# work
if(phase == "loc"){
  func_locTime(numRuns, phase, subjStr, dataDir, outDir)
}else if(phase == "task"){
  func_studyTime(numRuns, phase, subjStr, dataDir, outDir)
}

