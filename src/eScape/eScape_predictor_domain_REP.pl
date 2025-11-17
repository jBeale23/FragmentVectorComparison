#!/usr/bin/perl
#
# eScape_predictor.pl (adapted from linear model eScape v8.2.1)
# Read directory of FASTA files for prediction.
#
# sequences exists within a limited range of thermodynamic state 
# Thermodynamic Range taken from full ensembles under 
# Native and Denatured Conditions, respective to predictions.
#
# Predict dG, dHap, dHp, and TdS values
#
# 1) Random Selection of energetic values from *FULL* ensemble distributions
#    with WINDOW CONTEXT 
#
#  eScapeBlend_FULL contains the following:
#  Input features: (16 Features) Drawn randomly from full ensemble to predict target features.
#
#  dGN, ddGN, dHapN, ddHapN, dHpN, ddHpN, TdSN, dTdSN
#  dGD, ddGD, dHapD, ddHapD, dHpD, ddHpD, TdSD, dTdSD
#  * where ddGN, etc. correspond to changes observed with full ensemble.
#
#  Target features: (12 Features)
#  dGFN, dHapFN, dHpFN, TdSFN, dGF, dHapF, dHpF, TdSF, dGFD, dHapFD, dHpFD, TdSFD
#
#  want:  eScapeBlend(input) -> Real(Target) [Not eBlend Target]
#
# JGU 8/16/07 
# First version of working code completed.
#
# JGU 8/24/07
# incorporated strategies to handle missing data point not in library. Performance/impact needs to be tested.
#
# JGU 10/29/07
# LINEAR MODEL FOR W1.5 is FIXED
# Spiking data can be found in eScape_predictor_spike.pl
#
# JGU 11/30/07
# ADDED DOMAIN AVERAGING FEATURE
# $winsize (1, 5, 11, 15, 21, 31, 51, 81)
#
# JDB 6/12/25
# Added FindBin to automatically locate modules/libraries
#
# JDB 6/19/25
# Added Parallel::ForkManager to distribute the work of processing input fasta files across all available cpus
# 

use strict;
use FindBin;
use lib "$FindBin::Bin/modules";
use AAmisc;
use Algorithms;
use ArrayMisc;
use BasicStat;
use WinGrab;

# LINEAR MODEL: eScape_v8

# eBLEND LIBRARIES NEEDED TO GENERATE INPUT FEATURE
# files contain:
#
# 'dGN', 'ddGN', 'dHapN', 'ddHapN', 'dHpN', 'ddHpN', 'TdSN', 'dTdSN', 
# 'dGD', 'ddGD', 'dHapD', 'ddHapD', 'dHpD', 'ddHpD', 'TdSD', 'dTdSD', 
# 'dGFN', 'dHapFN', 'dHpFN', 'TdSFN', 'dGF', 'dHapF', 'dHpF', 'TdSF', 
# 'dGFD', 'dHapFD', 'dHpFD', 'TdSFD'
#
#  eScapeBlend_FULL contains values extracted from full ensemble.  
#  For example:  dGN = dGFN
#
#  eScapeBlend contains values extracted from respective ensemble only.
#
my $TVWIN_LIBPATH = "$FindBin::Bin/library";
my $TV_file1 = "eScapeBlend_FULL.features";
my $TV_file2 = "eScapeBlend_UNK.features";	# FOR UNKNOWN WINDOW
my $TV_file3 = "eScapeBlend_NULL.features";	# FOR UNNATURAL AMMINO ACIDS

my %TV_LIB = read_BLEND($TVWIN_LIBPATH, $TV_file1, $TV_file2, $TV_file3);
my @TV = ('dGN', 'dHapN', 'dHpN', 'TdSN', 'dGD', 'dHapD', 'dHpD', 'TdSD');

if ($#ARGV != 1) { die "eScape_predictor.pl [INDIR] [OUTDIR] \n"; }
my $INDIR = $ARGV[0];
my $OUTDIR = $ARGV[1];

use File::Path qw(make_path);
make_path($OUTDIR) unless -d $OUTDIR;

# Checks how many cpus are available and uses that as the number of processes to spawn
# If slurm is active, it uses the set number of cpus for the job rather than those available on the system
my $availCpu = $ENV{'SLURM_CPUS_PER_TASK'} || `getconf _NPROCESSORS_ONLN`;
use Parallel::ForkManager;
my $pm = Parallel::ForkManager->new($availCpu);

# Globs all input fasta files and distributes their processing across the number of available cpus
my @files = glob("$INDIR/*.fasta");
my @chunks = chunk_array(\@files, 4*$availCpu);


for my $chunk_ref (@chunks) {
	$pm->start and next;
	foreach my $file (@$chunk_ref) {
		my $log = $file; $log =~ s#fasta#eScapev8_domain_log#;
		$log =~ s#$INDIR#$OUTDIR#;

		my @FASTA = AAmisc::readFASTA($file);
		my @PRED;

			## COLLECTING INPUT FEATURES.
			open (LOG, ">$log") || die "Cannot write to LOG $log. \n";
		my @INPUTseq = getINPUT_Blend(\@FASTA);
		close(LOG);

		# print "CHECK INPUTseq should have 16 cols: $#{$INPUTseq[0]}. \n";
		
		# INPUTseq contains input features for the sequence where rows 
		# correspond to TVtype (min and max) and columns is residue position.
		# for Native and Denatured conditions, respectively.

		## GET PREDICTIONS:
		# Keys for index (min/max columns)

		my $pred;
			my %INDEX;
					foreach (0,1) { $INDEX{$_} = 'dGN'; }
					foreach (2,3) { $INDEX{$_} = 'dHapN'; }
					foreach (4,5) { $INDEX{$_} = 'dHpN'; }
					foreach (6,7) { $INDEX{$_} = 'TdSN'; }
					foreach (8,9) { $INDEX{$_} = 'dGD'; }
					foreach (10,11) { $INDEX{$_} = 'dHapD'; }
					foreach (12,13) { $INDEX{$_} = 'dHpD'; }
					foreach (14,15) { $INDEX{$_} = 'TdSD'; }

		foreach (@INPUTseq) {
		   my @INPUT = @{$_};
		   my %INPUT;
		   # RESHUFFLE ARRAY FOR PREDICTION INTO HASH FOR EACH TV
		   # HASH IS RENEWED FOR EACH POSITION.

		   for my $i (0 .. $#INPUT) {
			my $TVkey = $INDEX{$i};
			push(@{$INPUT{$TVkey}}, $INPUT[$i]);
	   } 	

	   # Make PREDICTIONS for EACH TV AND ENSEMBLE ( 16 INPUT Total )
	   my @PREDtmp;
	   foreach (@TV) {
	       my $TVkey = $_;
	       my ($min, $max) = @{$INPUT{$TVkey}};

	       # print "$TVkey $min $max \n";
	       $pred = eScape_v8($TVkey, $min, $max);
	       push (@PREDtmp, $pred);
	   }
	   push(@PRED, [@PREDtmp]);	
		} # FINISH FASTA


		### AVERAGE FOR "DOMAIN" PREDICTION
		# To capture collective contribution of each region to the ensemble energetics
		
		# TRANSFORM @PRED to calculate averages
		@PRED = ArrayMisc::transposeAoA(\@PRED);

		# AVERAGE over varying window size
		foreach my $winsize (1, 5, 11, 15, 21, 31, 51, 81) {
		   my %PRED;	
		   my (@OUT1, @OUT2);

			foreach (@PRED) {
			   my @ipred = @{$_};
			   %PRED = WinGrab::BasicStatWIN(\@ipred, $winsize);
				   push(@OUT1, $PRED{'AVG'});
			   push(@OUT2, $PRED{'SD'});
			} 

		   @OUT1 = ArrayMisc::transposeAoA(\@OUT1);
		   @OUT2 = ArrayMisc::transposeAoA(\@OUT2);

		   my $out1 = $file; $out1 =~ s#fasta#eScapev8_domain$winsize\_avgpred#;
			   my $out2 = $file; $out2 =~ s#fasta#eScapev8_domain$winsize\_sdpred#;
			   $out1 =~ s#$INDIR#$OUTDIR#;
		   $out2 =~ s#$INDIR#$OUTDIR#;

		   ArrayMisc::printAoA2file($out1, \@OUT1);
		   ArrayMisc::printAoA2file($out2, \@OUT2);
		}  # END DOMAIN AVERAGING AND OUTPUT
	}
	$pm-> finish;

} # END QUERY:
$pm->wait_all_children;

#### ================== END OF MAIN

#### ================== START SUBROUTINES 
# --- Helper function for splitting the workload across cpus ---
sub chunk_array {
    my ($array_ref, $chunks) = @_;
    my @array = @$array_ref;
    my @result;

    for (my $i = 0; $i < $chunks; $i++) {
        $result[$i] = [];
    }

    for my $i (0 .. $#array) {
        push @{ $result[$i % $chunks] }, $array[$i];
    }

    return @result;
} # END SUBROUTINE chunk_array
sub eScape_v8 {
        my ($TS, $min, $max) = @_;
        my $pred;

	if ($TS =~ /dGN/) {
	   $pred = ((0.8195 * $min) + (0.7492 * $max)) + 4696;
	}
	elsif ($TS =~ /dHapN/) {
	   $pred = ((0.7665 * $min) + (0.7632 * $max)) - 5068;
	}
	elsif ($TS =~ /dHpN/) {
	   $pred = ((0.7791 * $min) + (0.7524 * $max)) + 6195;
	}
	elsif ($TS =~ /TdSN/) {
	   $pred = ((0.7047 * $min) + (0.7507 * $max)) + 1998;
	}
        elsif ($TS =~ /dGD/) {
           $pred = ((0.7023 * $min) + (0.7479 * $max)) - 4154;
        }
        elsif ($TS =~ /dHapD/) {
           $pred = ((0.7801 * $min) + (0.6708 * $max)) + 447.53;
        }
        elsif ($TS =~ /dHpD/) {
           $pred = ((0.7115 * $min) + (0.8047 * $max)) - 94.37;
        }
        elsif ($TS =~ /TdSD/) {
           $pred = ((0.7319 * $min) + (0.6782 * $max)) + 3872 ;
        }
	else {
	   die "ERROR: Not a Thermodynamic State for Prediction. \n";
	}
	
	return $pred;

}  # END SUBROUTINE eScape_v8

sub getINPUT_Blend {
	my ($aref) = @_ ;
	my @FASTA = @{$aref};

	my @AAwin = WinGrab::SegmentizeAjoin(3, \@FASTA);

	my @INPUTseq;
	my @REPEAT;

	my $count = 0;
	for my $i (0 .. $#FASTA) {
	
		# PULL AAwin VALUES FROM LIBRARY
		my $AAwin = $AAwin[$i];
		my $AAunk = $AAwin;
			if ($AAunk =~ /.(.)./) {
				my $AA = $1;
				$AAunk = join('', (0, $AA, 0));
			}
		my @values;

		# print "KEY: $AAwin \n";
		if (defined $TV_LIB{$AAwin}) {
		   @values = @{$TV_LIB{$AAwin}};
		}
		elsif ($i == 0) {
                   $count++;
                   @values = @{$TV_LIB{$AAunk}};
                   print LOG "Using UNK for $AAwin at index: $i count: $count\n";
                }
		else {
		   $count++;
		   @values = @REPEAT;
		   print LOG "Using REPEAT vector for $AAwin at index: $i count: $count\n";
		}
                my @TVvalues = @values;
                @TVvalues = ArrayMisc::transposeAoA(\@TVvalues);
		@REPEAT = @values;

		# PULL RANGE FOR ALL FOUR TV for Denatured and Native Conditions	
                # INDEX TO eBLEND LIB:
                # (16)'dGFN', (17)'dHapFN', (18)'dHpFN', (19)'TdSFN'
                # (24)'dGFD', (25)'dHapFD', (26)'dHpFD', (27)'TdSFD'

		my @iTV = (16, 17, 18, 19, 24, 25, 26, 27);
		my @tmp_input;
		
		foreach my $i (@iTV) {	
		   my @TVrange = @{$TVvalues[$i]};
                   
		   # FIND MAX and MIN in VALUES
		   my ($TVmin, $TVmax) = Algorithms::minmax(\@TVrange);

		   # ADD VALUES OF EXTREME ENDS;
		   push(@tmp_input, $TVmin);
		   push(@tmp_input, $TVmax);
		}

		push(@INPUTseq, [@tmp_input]);
	}  # FIND MIN AND MAX FOR EACH RESIDUE (16 VALUES)

	#  AVERAGING WINDOWS for INPUT FEATURES 
	my $winsize = 5;

	@INPUTseq = ArrayMisc::transposeAoA(\@INPUTseq);
	# EACH ROW CORRESPND TO TV(min and max)
	for my $i (0 .. $#INPUTseq) {
		my @avgtmp = @{$INPUTseq[$i]};
		@avgtmp = WinGrab::avgWIN(\@avgtmp, $winsize);  
		@{$INPUTseq[$i]} = @avgtmp;
	}
	@INPUTseq = ArrayMisc::transposeAoA(\@INPUTseq);
	# ROW: Residue position

	return @INPUTseq;

}  # END SUBROUTINE getINPUT_Blend

sub read_BLEND {
        my ($lib, $file1, $file2, $file3) = @_;
        my %DATA;

	foreach my $fn ($file1, $file2, $file3) {
	   my $file = "$lib/$fn";
           open(FILE, $file) || die "Cannot open $file. \n";

           my $key;
           while (<FILE>) {
              if (/^[#\s]*([0A-Z]{3})/) {
                $key = $1;
              }
              elsif (/\d+/ && $key =~ /[A-Z0]{3}/) {
                my $line = $_;
                chomp($line);

                my @tmp = split('\t', $line);
                if ($tmp[$#tmp] eq '\s') {pop(@tmp);};
                push(@{$DATA{$key}}, [@tmp]);
              }
              else {
                next;
              }
           }
           close(FILE);
	}

        return %DATA;

}  # END SUBROUTINE read_BLEND
