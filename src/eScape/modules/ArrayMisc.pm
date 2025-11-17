package ArrayMisc;

use strict;

sub AoAoA2AoA {
	my ($aref) = @_;
	my @array1 = @{$aref};
	my @array2;

	# print "(1) $#array1 (2) $#{$array1[0]} (3) $#{$array1[0][0]} \n";

	foreach (@array1) {
		my @tmp = @{$_};
		# print "$tmp[0][0] \n";

		foreach (@tmp) {
		  push (@array2, $_);
		}
	}
	return(@array2);
}

sub diffAoA {
	my ($aref1, $aref2) = @_;
	my @a1 = @{$aref1};
	my @a2 = @{$aref2};
	my @a3;

	if ($#a1 != $#a2) {die "Array sizes are not equal for ArrayMisc::diffAoA. \n";}

	for my $i (0 .. $#a1) {
	if ($#{$a1[$i]} != $#{$a2[$i]}) {die "Array sizes are not equal for ArrayMisc::diffAoA. \n";}

		for my $j (0 .. $#{$a1[$i]}) {
			$a3[$i][$j] = $a1[$i][$j] - $a2[$i][$j];
		}
	}

	return @a3;
}

sub sumAoA {
        my ($aref1, $aref2) = @_;
        my @a1 = @{$aref1};
        my @a2 = @{$aref2};
        my @a3;

        if ($#a1 != $#a2) {die "Array sizes are not equal for ArrayMisc::sumAoA. \n";}

        for my $i (0 .. $#a1) {
        if ($#{$a1[$i]} != $#{$a2[$i]}) {die "Array sizes are not equal for ArrayMisc::sumAoA. \n";}

                for my $j (0 .. $#{$a1[$i]}) {
                        $a3[$i][$j] = $a1[$i][$j] + $a2[$i][$j];
                }
        }

        return @a3;
}

sub initializeAoAoA {
	my $size = $_[0];
	my @array;

	for my $i  (0 .. $size) {
	  for my $j ( 0 .. $size ) {
		$array[$i][$j] = ();	
	  }
	}

	return @array;
}

sub initializeAoA {
        my $size = $_[0];
        my @array;

        for my $i  (0 .. $size) {
                $array[$i] = ();
        }

        return @array;
}

sub cpAoA {
	my ($aref) = @_;
	my @AoA1 = @{$aref};	
	my @AoA2;

	for my $i (0 .. $#AoA1) {
		my @tmp = @{$AoA1[$i]};
		$AoA2[$i] = \@tmp;
	}
	return @AoA2;
}

sub hcatA {
        my ($aref1, $aref2) = @_;
        my @a1 = @{$aref1};
        my @a2 = @{$aref2};
        my @a;

        if ($#a1 != $#a2) { die "ArrayMisc::hcatAoA Arrays not eq in size. \n";}
        for my $i (0 .. $#a1) {
                push (@a, [($a1[$i], $a2[$i])]);
        }
        return @a;
}

sub hcatAoA {
	my ($aref1, $aref2) = @_;
	my @a1 = @{$aref1};
	my @a2 = @{$aref2};
	my @a;

	if ($#a1 != $#a2) { die "ArrayMisc::hcatAoA Arrays not eq in size. \n";}
	foreach (0 .. $#a1) {
		@{$a[$_]} = (@{$a1[$_]}, @{$a2[$_]});
	}
	return @a;
}

sub A2AoA {
   my ($aref) = @_;
   my @a1 = @{$aref};
   my @a2;

	for my $i ( 0 .. $#a1) {
		$a2[$i][0] = $a1[$i];	
	}

   return @a2;
}

# Convert AoA -> A;
sub Linearize {
	my ($aref) = @_;
	my @DATA = @{$aref};
	my @L;

	foreach (@DATA) {
		push (@L, @{$_});
	}
	return @L;
}

sub HoA2AoA {
  my ($href) = @_;
  my %DATA = %{$href};
  my @DATA;

  foreach my $key (keys %DATA) {
	my @tmp = @{$DATA{$key}};
	push(@DATA, [($key, @tmp)]);
  }

  return @DATA;
}


sub getCOL {
  my ($aref, $colN) = @_;
  my @DATA = @{$aref};
  my @col;

  for my $i ( 0 .. $#DATA) {
	push(@col, $DATA[$i][$colN]);
  } 
  return @col;
}

sub printA {
	# PRINT to STDOUT
        my @array = @{$_[0]} ;
	foreach (@array) {
		print "$_\n";
	}	
}

sub printA2file {
        my ($file, $aref) = @_;
        my @array = @{$aref};
        open (OUT, ">$file") || die "Cannot open $file in printA2file. \n";
       
        foreach (@array) {
                print OUT "$_\n";
        }
	close(OUT);

}


sub printAoA {
        # PRINT to STDOUT

        my @array = @{ $_[0] } ;
        for my $i (0 .. $#array) {
		for my $j (0 .. $#{$array[$i]}) {
                	print "$array[$i][$j]\t";
		}
		print "\n";
        }
}

#usage: ArrayMisc::printAoA2file($out, \@DATA);
sub printAoA2file {
        # PRINT to file

        my ($file, $aref) = @_;

	my @array = @{$aref};
	open (OUT, ">$file") || die "Cannot open $file in printAoA2file. \n";
        for my $i (0 .. $#array) {
                for my $j (0 .. $#{$array[$i]}) {
                        print OUT "$array[$i][$j]\t";
                }
                print OUT "\n";
        }
	close(OUT);
}

sub printAoAoA {
        # PRINT to STDOUT

        my @array = @{ $_[0] } ;
        for my $i (0 .. $#array) {
		print "$i \n";
                for my $j (0 .. $#{$array[$i]}) {
		     for my $k (0 .. $#{$array[$i][$j]}) {
                        print "$array[$i][$j][$k]\t";
		     }
		     print "\n";
                }
                print "\n";
        }
}

#usage: ArrayMisc::printAoA2file($out, \@DATA);
sub printAoAoA2file {
        # PRINT to file

        my ($file, $aref) = @_;

        my @array = @{$aref};
        open (OUT, ">$file") || die "Cannot open $file in printAoA2file. \n";
        for my $i (0 .. $#array) {
		print OUT "$i \n";
                for my $j (0 .. $#{$array[$i]}) {
		    for my $k (0 .. $#{$array[$i][$j]}) {
                        print OUT "$array[$i][$j][$k]\t";
		    }
		    print OUT "\n";
                }
                print OUT "\n";
        }
        close(OUT);
}

sub convert2AoA {
	my @array = @{ $_[0] };

	for my $i (0 .. $#array) {
        	my @tmp = split('\s+', $array[$i]);
        		if ($tmp[0] eq '') {
                		shift(@tmp);
        	}
        	$array[$i] = [@tmp];
	}
	return @array;
}

sub readA {
	my $file = $_[0];
	open (IN, $file) || die "Cannot open $file in ArrayMisc::readA\n";
	my @out = <IN>;
	chomp(@out);
	close(IN);
	return @out;	
}

sub readAoA {
	my $file = $_[0] ;
        open (IN, $file) ; #|| die "Cannot open $file in ArrayMisc::readAoA \n"; 
      	my @out = <IN>;
	chomp(@out);
	if ($#out == -1) {print "Cannot open $file in ArrayMisc::readAoA \n";}
	close(IN);
	@out = convert2AoA(\@out);
        return @out;
}


sub readAoAtab {
	my $file = $_[0];
	open (IN, $file);
	my @out = <IN>;
	chomp(@out);
	close(IN);

	my @OUT;
	foreach (@out) {
		my @tmp = split('\t', $_);
		# print "TMP : $#tmp \n $tmp[0]";
		push (@OUT, [@tmp]);
	}

	return(@OUT);
}

sub readRtable {
	my $file =$_[0];
	my @out;
	my $flag = 0;

	open(IN, $file) || die "Cannot open R file to read table. \n";
	while (<IN>) {
	   if ($flag == 0) {
		$flag = 1;
		my $line = $_;
		chomp($line);

		my @tmp = split('\s+', $line);
		push(@out, [@tmp]);
	   }
	   else {
		my $line = $_;
		chomp($line);

		my @tmp = split('\s', $line);
		shift(@tmp);
		push(@out, [@tmp]);
	   }
	}
	return @out;
}

sub sort {
    my ($type, $aref) = @_;
    my @a = @{$aref}; 
    my @sorted;

    if ($type eq 'asc') {
	@sorted = sort {$a <=> $b} @a;
    }
    elsif ($type eq 'des') {
	@sorted = sort {$b <=> $a} @a;
    }
    elsif ($type eq 'alpha') {
	@sorted = sort{ lc($a) cmp lc($b) } @a;
    }
    else {
	die "Error in sort option. \n";
    }

    return @sorted;
}

sub sortbyCOL {
    my ($aref, $col, $type) = @_;

    my @iSORT;
    my @SORT;
    my @DATA = @{$aref};
    my @COL = getCOL(\@DATA, $col);
    
   # print "Number of DATA: $#DATA \n";
   # print "Number of COLDATA: $#COL\n"; 
   # SORT BY VALUE
   if ($type eq 'desc') {
	@iSORT = sort{ $COL[$b] <=> $COL[$a]} (0 .. $#COL);
   }
   elsif ($type eq 'asce') {
	@iSORT = sort{ $COL[$a] <=> $COL[$b]} (0 .. $#COL);
   }
   elsif ($type eq 'alpha') {
	@iSORT = sort{ $COL[$a] cmp $COL[$b]} (0 .. $#COL);
   }
   else {die "Wrong type. \n";}

   foreach my $i (@iSORT) {
	push (@SORT, $DATA[$i]);
   }

   return @SORT;
}

sub sortHkey {
    my ($type, $href) = @_;
    my %hash = %{$href};
    my @sorted;
        
    if ($type eq 'value') {
        @sorted = sort {$hash{$a} cmp $hash{$b} } keys %hash;
    }
    else {
        die "Error in sortHkey option. \n";
    }

    return @sorted;
}

sub sortindex {
    my ($aref, $type) = @_;

    my @iSORT;
    my @SORT;
    my @DATA = @{$aref};

   # print "Number of DATA: $#DATA \n";
   # print "Number of COLDATA: $#COL\n";
   # SORT BY VALUE

   if ($type eq 'desc') {
        @iSORT = sort{ $DATA[$b] <=> $DATA[$a]} (0 .. $#DATA);
   }
   elsif ($type eq 'asce') {
        @iSORT = sort{ $DATA[$a] <=> $DATA[$b]} (0 .. $#DATA);
   }
   else {die "Wrong type. \n";}

   return @iSORT;
}

sub transposeAoA {
	my ($aref) = @_;
	my @a1 = @{$aref};
	my @a2;

	for my $i (0 .. $#a1) {
		for my $j (0 .. $#{$a1[$i]}) {
			if ($a1[$i][$j] ne '') {
			$a2[$j][$i] = $a1[$i][$j];
			}
		}
	}

	return @a2;
}

sub readH {
	my ($file) = @_;
	my @tmp = readAoA($file);
	my %DATA;

	my($key, $values);
	foreach (@tmp) {
	  my @tmp1 = @{$_};
	  $key = shift(@tmp1);

	  if ($#tmp1 == -1) {$values = ();} 
	  elsif ($#tmp1 == 0) {$values = shift(@tmp1); }
	  else {
	  	$values = \@tmp1;
	  }
	  
	  $DATA{$key} = $values;
	}

	return %DATA;
}

sub readHoA {
	# FIRST COLUMN IS KEY

        my ($file) = @_;
        my @tmp = readAoA($file);
        
	my %DATA;
        my($key, @values);
        foreach (@tmp) {
          my @tmp1 = @{$_};
          $key = shift(@tmp1);
	  @values = @tmp1;

          $DATA{$key} = [@values];
        }

        return %DATA;
}

sub readHoA_HEAD {

        # ASSUMES KEY IS AT HEAD OF AoA;

        my ($file) = @_;
        my @tmp = readAoA($file);
        @tmp = transposeAoA(\@tmp);

        my %DATA;
        my($key, @values);
        foreach (@tmp) {
          my @tmp1 = @{$_};
          $key = shift(@tmp1);
          @values = @tmp1;

          $DATA{$key} = [@values];
        }

        return %DATA;
}
sub printH {
	my %DATA = %{ $_[0] };
	foreach( keys %DATA ) {
		print "$_ \t $DATA{$_} \n";
	}
}

sub printH2file {
	my ($file, $href) = @_;
        my %DATA = %{ $href };
	
	open (OUT, ">$file") || die "Cannot open $file in printH2file";
        foreach( keys %DATA ) {
                print OUT "$_ \t $DATA{$_} \n";
        }
	close(OUT);
}

sub printHkeys2file {
        my ($file, $href) = @_;
        my %DATA = %{ $href };

        open (OUT, ">$file") || die "Cannot open $file in printH2file";
        foreach( keys %DATA ) {
                print OUT "$_ \n";
        }
        close(OUT);
}

sub printHoH2file {
        my ($file, $href) = @_;
        my %DATA = %{ $href };

        open (OUT, ">$file") || die "Cannot open $file in printH2file";
	print OUT "ID1\t ID2\t DATA\n";

        foreach my $id ( keys %DATA ) {
		foreach my $jd (keys %{$DATA{$id}}) {
                	print OUT "$id\t $jd\t $DATA{$id}{$jd}\n";
		}
        }
        close(OUT);
}

sub printHoA2file {
	my ($file, $href) = @_;
	my %DATA = %{ $href };

	open (OUT, ">$file") || die "Cannot open $file in printHoA2file\n";
	
	foreach my $id (keys %DATA) {
		print OUT "$id";
		for my $i (0 .. $#{$DATA{$id}}) {
			print OUT "\t$DATA{$id}[$i]";
		}
		print OUT "\n";
	}
	close(OUT);	
}

sub printHoAoA2file {
        my ($file, $href) = @_;
        my %DATA = %{ $href };

        open (OUT, ">$file") || die "Cannot write to $file in printHoAoA2file\n";

        foreach my $id (keys %DATA) {
                print OUT "$id\n";
                for my $i (0 .. $#{$DATA{$id}}) {
		   for my $j (0 .. $#{$DATA{$id}[$i]}) {
                        print OUT "$DATA{$id}[$i][$j]\t";
		   }
		   print OUT "\n";
                }
                print OUT "\n";
		# print "KEY: $id VALUE: $DATA{$id} \n";
        }
        close(OUT);
}

sub shuffleA {
    my $array = shift;
    my $i;
    for ($i = @$array; --$i; ) {
        my $j = int rand ($i+1);
        next if $i == $j;
        @$array[$i,$j] = @$array[$j,$i];
    }

    return @$array;
}

1;
