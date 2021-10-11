
clc;
clear all;

% Path of libsvm
addpath('C:\Users\libsvm-3.24\matlab');

% Nine problems
pros=[1 2 3 4 5 6 7 8 9];

% Grid points for MS
c_range=-6:2:12;
g_range=-6:2:12;

n_c=length(c_range);
n_g=length(g_range);

for real_pro=1:length(pros)  
    i_pro=pros(real_pro);
    data_num=i_pro;

    load(['feature' num2str(data_num) '.mat']);
    
    results=zeros(n_c,n_g,n_sub); % n_sub = # of subjects

    % Consider LOSOCV validation: test=data of one subject / training=data
    % of rest of subjects
    for ii_sub=1:n_sub
        % For normalization
        training=[];
        for i_target=1:n_sub
            if (i_target~=ii_sub)
                nn=features(1,(i_target-1)*(n_feature+1)+1);
                pre_training=features(2:nn+1,(i_target-1)*(n_feature+1)+1:(i_target)*(n_feature+1));
                training=[training; pre_training];
            end
        end
        data=training(:,1:n_feature);
        min_vec=min(data,[],1);
        max_vec=max(data,[],1);
        data=[];
        
        % Calculate LOSOCV-based performance measure
        for i_sub=1:n_sub
            if ii_sub~=i_sub
                %%% Data preparation
                n_test=features(1,(i_sub-1)*(n_feature+1)+1);
                test_data=features(2:n_test+1,(i_sub-1)*(n_feature+1)+1:(i_sub)*(n_feature+1));
                training=[];
                
                for i_data=1:n_sub
                    if (i_data~=i_sub) && (i_data~=ii_sub)
                        i_target=i_data;

                        nn=features(1,(i_target-1)*(n_feature+1)+1);
                        pre_training=features(2:nn+1,(i_target-1)*(n_feature+1)+1:(i_target)*(n_feature+1));
                        training=[training; pre_training];
                    end
                end

                pre_training=[];
                training(:,1:n_feature)=(training(:,1:n_feature) - repmat(min_vec,size(training(:,1:n_feature),1),1))*spdiags(1./(max_vec-min_vec)',0,size(training(:,1:n_feature),2),size(training(:,1:n_feature),2));
                test_data(:,1:n_feature)=(test_data(:,1:n_feature) - repmat(min_vec,size(test_data(:,1:n_feature),1),1))*spdiags(1./(max_vec-min_vec)',0,size(test_data(:,1:n_feature),2),size(test_data(:,1:n_feature),2));

                % Calculate LOSOCV-based performance measure for each grid
                % points
                for log2c = 1:n_c
                   c=c_range(log2c);
                  for log2g = 1:n_g
                    g=g_range(log2g);
                    cmd = ['-s 0 -t 2 -c ', num2str((2^c)), ' -g ', num2str(2^g)];
                    model = svmtrain(training(:,n_feature+1), sparse(training(:,1:n_feature)), cmd);

                    [~,acc,~] = svmpredict(test_data(:,n_feature+1), sparse(test_data(:,1:n_feature)), model);
                    results(log2c,log2g,ii_sub)=results(log2c,log2g,ii_sub)+acc(1);

%                      clc
                  end
                end
            end
        end
    end
    
    results=results/(n_sub-1)
end
