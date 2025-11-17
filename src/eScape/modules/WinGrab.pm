package WinGrab;
use strict;
use lib '/Users/jgu/Research/Software/perl/modules/';
use BasicStat;
use POSIX;

sub avgWIN {
  my ($aref, $winsize) = @_;
  my @DATA = @{$aref};
  my @AVG = ();

  my @WINS = SegmentizeA($winsize, \@DATA);
  @WINS = zero2null_AoA(\@WINS);

  foreach (@WINS) {
	my $Waref = $_;
	my @SUB = @{$Waref};
	my $mean = BasicStat::mean(\@SUB);
	push(@AVG, $mean);
  }     
  return @AVG;
}

sub BasicStatWIN {
  my ($aref, $winsize) = @_;
  my @DATA = @{$aref};
  my %DATA; 

  $DATA{'AVG'} = ();
  $DATA{'SD'} = ();
	
  my @WINS = SegmentizeA($winsize, \@DATA);
  @WINS = zero2null_AoA(\@WINS);

  foreach (@WINS) {
        my $Waref = $_;
        my @SUB = @{$Waref};
        my $mean = BasicStat::mean(\@SUB);
	my $sd = BasicStat::stdev($mean, \@SUB);
        push(@{$DATA{'AVG'}}, $mean);
	push(@{$DATA{'SD'}}, $sd);
  } 
  return %DATA;
}

sub get_seg {
	# Grabbing sequence within window size.
	# FOR AoA;
	my ($pos, $win, $aref) = @_;
	my @a = @{ $aref } ;	# ENTIRE LENGTH OF SEQUENCE
	my @seg;
	my @zeros;
		
                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE
		# POTENTIAL BUG:
		# Might want to reconsider using zero matrix
		# for @seg = (@seg, 0);

                my @win = get_win($win);
	
		my $size = $#{$a[0]};
		for my $i (0 .. $size) {
			$zeros[$i] = 0;
		}	

                foreach (0 .. $#win) {
                        my $spos = $pos + $win[$_];

                        if ($spos < 0 || $spos> $#a) {
                                @seg = (@seg, @zeros);
                        }
                        else {
                                @seg = (@seg, @{$a[$pos]});
                        }
                }
		return @seg;
}

sub get_segA_beg {
        # Grabbing sequence within window size.
        my ($pos, $win, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE
        my @seg;
        my @zeros;

	my $translate = $win/2;
	$translate = floor($translate);
	$pos = $pos +$translate;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE
                # POTENTIAL BUG:
                # Might want to reconsider using zero matrix
                # for @seg = (@seg, 0);

                my @win = get_win($win);
		
                for my $i (0 .. $#win) {
                        $zeros[$i] = 0;
                }

                foreach (0 .. $#win) {
                        my $spos = $pos + $win[$_];

                        if ($spos < 0 || $spos> $#a) {
                                @seg = (@seg, @zeros);
                        }
                        else {
                                @seg = (@seg, $a[$pos]);
                        }
                }
                return @seg;
}


# WinGrab::Segmentize($win, \@DATA);
sub Segmentize {
        # Grabbing sequence within window size.
        my ($win, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE

	my @Aseg; 
        my @zeros;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE

                my @win = get_win($win);
		
		for my $i (0 .. $#{$a[0]}){
                        $zeros[$i] = 0;
                }

		for my $i (0 .. $#a) {
			my @seg = ();
                	foreach (0 .. $#win) {
                       		my $pos = $i + $win[$_];

                        	if ($pos < 0 || $pos> $#a) {
                               		@seg = (@seg, @zeros);
                        	}
                        	else {
                                	@seg = (@seg, @{$a[$pos]});
                        	}
                	}
			$Aseg[$i] = [@seg];
		}
		return @Aseg;
}

# USE THIS FOR SINGLE SEQUENCES (one array)
sub SegmentizeA {
        # Grabbing sequence within window size.
        my ($win, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE

        my @Aseg;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE

                my @win = get_win($win);

                for my $i (0 .. $#a) {
                        my @seg = ();
                        foreach (0 .. $#win) {
                                my $pos = $i + $win[$_];

                                if ($pos < 0 || $pos> $#a) {
                                        @seg = (@seg, 0);
                                }
                                else {
                                        @seg = (@seg, $a[$pos]);
                                }
                        }
                        $Aseg[$i] = [@seg];
                }
                return @Aseg;
}

sub SegmentizeAjoin {
        # Grabbing sequence within window size.
        my ($win, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE

        my @Aseg;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE

                my @win = get_win($win);

                for my $i (0 .. $#a) {
                        my @seg = ();
                        foreach (0 .. $#win) {
                                my $pos = $i + $win[$_];

                                if ($pos < 0 || $pos> $#a) {
                                        @seg = (@seg, 0);
                                }
                                else {
                                        @seg = (@seg, $a[$pos]);
                                }
                        }
                        $Aseg[$i] = [@seg];
                }

		my @Ajoin;
		for my $i (0 .. $#Aseg) {
		  $Ajoin[$i] = join('', @{$Aseg[$i]});
		}
                return @Ajoin;
}

sub Segmentizeref {
        # Grabbing sequence within window size.
        my ($win1, $win2, $aref) = @_;
        my @a = @{ $aref } ;    # ENTIRE LENGTH OF SEQUENCE

        my @Aseg;
        my @zeros;

                ## CREATING SVM INPUT DATA FOR GIVEN POS AND WINDOW SIZE

                my @win1 = get_win($win1);
		my @win2 = get_win($win2);

                for my $i (0 .. $#{$a[0]}){
                        $zeros[$i] = 0;
                }

                for my $i (0 .. $#a) {
                        my @seg = ();
                        for my $w1 (0 .. $#win1) {
		   	    my $pos1 = $i + $win1[$w1];
	
			    for my $w2 (0 .. $#win2) {
                                my $pos2 = $pos1 + $win2[$w2];

                                if ($pos2 < 0 || $pos2> $#a) {
                                        push(@{$seg[$w1]}, @zeros);
                                }
                                else {
                                        push(@{$seg[$w1]}, @{$a[$pos2]});
                                }
			    }
                        }
                        $Aseg[$i] = [@seg];
                }
                return @Aseg;
}

sub get_win {
	# Create index array for grabbing a window of sequence.
	
        my ($win) = @_;
        my @win;

                if ($win == 81) {
                     my $start = -40;
                     for (1 .. 81) {
                        push (@win, $start);
                        $start = $start + 1;
                     }
                }
                elsif ($win == 51) {
                     my $start = -25;
                     for (1 .. 51) {
                        push (@win, $start);
                        $start = $start + 1;
                     }               
		}
		elsif ($win == 41) {
		     my $start = -20;
                     for (1 .. 41) {
                        push (@win, $start);
                        $start = $start + 1;
                     }
		}
		elsif ($win == 31) {
		     my $start = -15;
		     for (1 .. 31) {
			push (@win, $start);	
			$start = $start + 1;
		     }	
		}
		elsif ($win == 21) {
                        @win = (-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10);
                }	
		elsif ($win == 19) {
                        @win = (-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9);
		}	
		elsif ($win == 17) {
			@win = (-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8);
		}
                elsif ($win == 15) {
                        @win = (-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7);
                }
                elsif ($win == 13) {
                        @win = (-6, -5,-4,-3,-2,-1,0,1,2,3,4,5,6);
                }
                elsif ($win == 11) {
                        @win = (-5,-4,-3,-2,-1,0,1,2,3,4,5);
                }
                elsif ($win == 9) {
                        @win = (-4,-3,-2,-1,0,1,2,3,4);
                }
                elsif ($win == 7) {
                        @win = (-3,-2,-1,0,1,2,3);
                }
                elsif ($win == 5) {
                        @win = (-2, -1, 0, 1, 2);
                }
		elsif ($win == 3) {
			@win = (-1, 0, 1);
		}
		elsif ($win == 1) {
			@win = (0);
		}
                else {
                        die "Window size not available try different or fix module:\n";
                }
                return @win;
}

sub zero2null_AoA {
	my ($aref) = @_;
	my @DATA = @{$aref};

	for my $i (0 .. $#DATA) {
		for my $j ( 0 .. $#{$DATA[$i]}) {
			if ($DATA[$i][$j] == 0) {
				$DATA[$i][$j] = '';
			}
		}
	}

	return @DATA;
}
1;
