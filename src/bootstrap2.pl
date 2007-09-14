# Created: David Madigan
# bootstrap2.pl: Alex Genkin, Nov'05
#	- keep the base model intact, not overwrite with boot models
#	- correct upper/lower case for Linux portability: 'BBRtrain'
#	- hyperparameter in case of Laplace is not the same as variance in -V; fixed: use with BBR ver>=2.61
#	- print mean/stddev for the intercept as well as for all other params
#	- no need to die when '-N 1' and no '-V': view '-N 1' as same hyperpar-defining strategy for each replicate

use strict;
srand(22);

# read the command line switches. -B nn specifies the number of bootstrap
# replicates (default=30); -N 1 (default)for new hyperparameter 
# every  boostrap sample, 0 to choose optimal on training data and reuse.
# Remaining command line options pass through to BBRtrain.
# The last two have to be the training data file and the model file
#
# perl bootstrap.pl -B 30 -N 0 -p 2 -C 10,10 -V 0.0001,0.001,0.01,0.1,1,10,100,1000 extract.bbr entity.mod
#

my (
    $temp, $i, $j, $k, $l, @temp, $foo,
    $n,                                       # training sample size
    $commandString, $bootCommandString,       # the command lines options
    $trainingFile, $modelFile,
    @sampleCount,                             # keeps counts for sampling with replacement
    @indices,
    @parameters,                              # stores the parameters for a single model
    @parameterList,                           # stores the parameters for all the models
    $intercept,                               # stores the intercept for a single model
    @intercepts,                              # stores the intercepts for all the models
    $priorVar,
    $numBootstrapReplicates,
    $newHyperEachReplicate,
    %featName
    );

$commandString = join " ",@ARGV;

if ($commandString =~ /-h/) {
    print "Here is how to use this program:\n\n";
    print "perl bootstrap.pl [options] training_data_file model_file\n\n";
    print "where the options are:\n\n";
    print "-B <integer>, Number of bootstrap replicates (default is 30)\n";
    print "-N <[0,1]>, 1-new hyperparam per replicate 0-same hyperparam per replicate (default is 1)\n\n";
    print "bootstrap.pl passes all other options through to BBRtrain.\n\n";
    print "Here's an example (laplace prior, cv for hyperparameter):\n\n";
    print "perl bootstrap.pl -B 30 -N 1 -p 1 -C 10,5 -V 0.1,1,10 trn.dat entity.mod\n\n\n";
    die;
}

# get the filenames
$commandString =~ / ([^ ]+) +([^ ]+) *$/;
$trainingFile = $1;
$modelFile = $2;

# get number of bootstrap replicates
$numBootstrapReplicates = 30;
if ($commandString =~ /-B (\d+)/) {
    if ($1 > 0) {
	$numBootstrapReplicates = $1;
    }
    $commandString =~ s/-B \d+//;    # excise that piece
} 

# get flag for reselecting hyperparameter every replicate
$newHyperEachReplicate = 1;
if ($commandString =~ /-N (\d)/) {
    if (($1==0)||($1==1)) {
	$newHyperEachReplicate = $1;
    }
    $commandString =~ s/-N \d//;    # excise that piece
} 

# kick out illogical combos
#--- AG: I don't see that necessary, vewing '-N 1' as same hyperpar-defining strategy for each replicate
#if (($newHyperEachReplicate == 1) && ($commandString !~ /-V /)) {
#    print "for new hyperparemter per replicate (i.e., -N 1) need to give candidates via -V\n\n";
#    die;
#}

# first generate the model file containing the actual point estimates
# and pick up the hyperparameter value
$temp = `BBRtrain $commandString`;
if ($temp =~ /Final prior variance value/) {   #AG: new unified BBR log format, starting ver 2.61
    $temp =~ /Final prior variance value (\S+)/;
    $priorVar = $1;
}
elsif ($commandString =~ /-C/) { #AG: this isn't quite correct: cv can run without -C
    $temp =~ /Best hyperparameter value (.*?) cv-average/;
    $priorVar = $1;    #AG: this is wrong with Laplace: hyperparameter!=prior var
}
else {
    $commandString =~ /-V ([\d\.]+) /;
    $priorVar = $1;
}

# if not picking a hyperparameter per replicate and candidates given, 
# need to pick one at the outset
if (($newHyperEachReplicate == 0) && ($commandString =~ /-V /)) {
    if (!($priorVar > 0)) {print "Need hyperparameter candidates\n"; die;}
    $commandString =~ s/-C [\d\,\.]+//;    # get rid of the cv command
    $commandString =~ s/--autosearch//;    # AG: get rid of the new autosearch command
    $commandString =~ s/-V [\d\,\.]+/-V $priorVar/;    # fix the hyperparameter
    printf "CV Prior var from training: %.2f\n",$priorVar;
}


######################################################################################
# main bootstrap loop
######################################################################################

$bootCommandString = $commandString;
$bootCommandString =~ s/$trainingFile/boot\.dat/;

# get the original sample size
$n=0;
open(TRAIN,$trainingFile);
while (<TRAIN>) {
    $n++;
}
close(TRAIN);

for ($i=0; $i <= $numBootstrapReplicates; $i++) {

    printf "rep: %.0f   prior variance: %.5f\n",$i,$priorVar;  #AG: print 'prior var', was 'hyperparameter'
    # First get the parameter estimates from the model file.
    # These will the training point estimates first time through.
    # Note that I execute this loop one extra time to pick up the
    # last of the bootstrap estimates.

    open(BAYESMODEL,$modelFile) or die "can't open model file $modelFile";
    do {
	$temp = <BAYESMODEL>;
	chomp($temp);
    } until (($temp =~ /^topicFeats/)||(!$temp));

    if (!$temp) {    # deals with the case where bbr produces no model
	$i--;
    }
    else {
	@indices =  split(" ",$temp);
	shift @indices;   # get rid of the word topicFeats
    
	do {
	    $temp = <BAYESMODEL>;
	} until ($temp =~ /^beta/);
	close(BAYESMODEL);
	
	chomp($temp);
	@parameters = split(" ",$temp);
	shift @parameters;
	
	$intercept = pop @parameters;
#    print "intercept\t",$intercept,"\n";
	push @intercepts,$intercept;
	
	# note: there can be missing values on the parameterList. This happens
	# when a predictor doesn't appear in a bag. Probably should be set to zero.
						#(AG: I don't understand this comment) 
	
	for ($j=0; $j < @parameters; $j++) {
	    $parameterList[$indices[$j]]->[$i] = $parameters[$j];
	}
    }

    if ($i==0) { #AG: need to keep the original model intact!
	$bootCommandString =~ s/$modelFile/boot\.model/;
	$modelFile = "boot\.model";
    }
    
    if ($i < $numBootstrapReplicates) {     # no need to do this last time around
	open(BOOT,">boot.dat");
	undef @sampleCount;
	for ($j = 1; $j <= $n; $j++) {
	    $sampleCount[1+ int($n * rand())]++;
	}
	$k = 0;
	open(TRAIN,$trainingFile);
	while (<TRAIN>) {
	    $k++;
	    for ($l = 1; $l <= $sampleCount[$k]; $l++) {
		print BOOT;
	    }
	}
	close(TRAIN);
	close(BOOT);
	
	$temp = `BBRtrain $bootCommandString`;
	if ($temp =~ /Final prior variance value/) {   #AG: new unified BBR log format, starting ver 2.61
	    $temp =~ /Final prior variance value (\S+)/;
	    $priorVar = $1;
	}
	elsif ($bootCommandString =~ /-C/) {
	    $temp =~ /Best hyperparameter value (.*?) cv-average/;
	    $priorVar = $1;
	}
	else {
	    $bootCommandString =~ /-V ([\d\.]+) /;
	    $priorVar = $1;
	}
    }
}


open(FEATURES,"features.bbr");
while (<FEATURES>) {
    chomp;
    @temp = split ":";
    $featName{$temp[0]} = $temp[1];
}


printf "Intercept\t%.2f ",$intercepts[0];
shift @intercepts;
my $stddev = sqrt(variance(\@intercepts));
printf("\t%.2f",$stddev);
# AG: added; treat intercept as other params
if ($stddev>0) {
  printf("\t%.2f\n",$intercepts[0]/$stddev);
}else{ 
  printf("\tNA\n");
}

for ($i=0; $i < @parameterList; $i++) {
    $temp = $parameterList[$i];
    if (defined $temp) {
	@temp = @$temp;
	print "$i";
	printf "\t$featName{$i}\t%.4f ",$temp[0];
	$foo = $temp[0];
	shift @temp;
	if (@temp > 1) {
	    $temp = sqrt(variance(\@temp));
	    if ($temp > 0) {
		printf "\t%.4f\t%.4f\n",$temp,$foo/$temp;
	    }
	    else {
		printf "\t%.4f\tNA\n",$temp;
	    }
	}
	else {
	    printf "\t%.4f\n",0;
	}
    }
}

sub variance {

    # pass this routine an array reference

    my (@x, @xsq, $x, $n, $i);

    $x =  shift;
    @x = @$x;
    for ($i = 0; $i < @x; $i++) {
	$xsq[$i] = $x[$i] * $x[$i];
    }
    $n = @x;

    if ($n <= 1) {return 0;}
    return ((sum(\@xsq) - ((sum(\@x)*sum(\@x)/$n)))/($n-1));
}

sub sum {

    # pass this routine an array reference

    my ($x, $temp, $total);
    $x =  shift;
   
    foreach $temp (@$x) {
	if (defined $temp) {
	    $total += $temp;
	}
    }
    return $total;
}




