       include 'matman.f'


       program matjoind

C -- Program MATJOIND combines two matrices into a composite matrix.

       use matrix_manipulate
       implicit none

       logical          :: lopened
       integer          :: ifail,ncol,nrow,mcol,mrow,ierr,jcol,jrow,
     +                     icol,irow
       integer          :: lw(5),rw(5)
       character*12     :: aversion
       character*20     :: aarow,aacol
       character*200    :: matfile1,matfile2,outfile
       character*200    :: afile
       character*500    :: cline
       character*600    :: amessage
       type(dmatrix)    :: mat1,mat2,mat3

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' MATJOIND version ',a,'. Watermark Numerical ',
     + 'Computing.')
#ifdef NO_CMDLINE
       write(*,*)
1      write(*,'(a)') ' ENTER COMMAND LINE ARGUMENTS > '
       read(*,'(a)',ERR=1) CLINE
#else
       call pgetcl(cline)
#endif

C -- The command line is parsed.

       if(cline.eq.' ') go to 8900
       call spacesub(cline)
       call linspl(ifail,4,lw,lw,cline)
       if(ifail.eq.0) go to 8900
       call linspl(ifail,3,lw,rw,cline)
       if(ifail.ne.0) go to 8900
       matfile1=cline(lw(1):rw(1))
       call remchar(matfile1,char(211))
       matfile2=cline(lw(2):rw(2))
       call remchar(matfile2,char(211))
       outfile=cline(lw(3):rw(3))
       call remchar(outfile,char(211))
#ifndef UNIX
       call lowcas(matfile1)
       call lowcas(matfile2)
       call lowcas(outfile)
#endif

C -- The two matrices are read.

       call mat_read(ifail,10,mat1,matfile1,amessage,cline)
       if(ifail.ne.0) go to 9800
       call addquote(matfile1,afile)
       write(6,100) trim(afile)
100    format(' - file ',a,' read ok.')

       call mat_read(ifail,10,mat2,matfile2,amessage,cline)
       if(ifail.ne.0) go to 9800
       call addquote(matfile2,afile)
       write(6,100) trim(afile)

C -- The two matrices are to ensure no overlap.

       if(mat1%icode.eq.-1)then
         call mat_diag_expand(ifail,mat1,amessage)
         if(ifail.gt.0) go to 9800
       end if
       if(mat2%icode.eq.-1)then
         call mat_diag_expand(ifail,mat2,amessage)
         if(ifail.gt.0) go to 9800
       end if
       if(mat1%icode.eq.1)then
         call mat_icode_expand(ifail,mat1,amessage)
         if(ifail.gt.0) go to 9800
       end if
       if(mat2%icode.eq.1)then
         call mat_icode_expand(ifail,mat2,amessage)
         if(ifail.gt.0) go to 9800
       end if

       do irow=1,mat1%nrow
         aarow=mat1%arow(irow)
         do jrow=1,mat2%nrow
           if(aarow.eq.mat2%arow(jrow)) then
             write(amessage,110)
110          format(' At least one row in the first matrix has the ',
     +       'same name as a row of the second matrix.')
             go to 9800
           end if
         end do
       end do

       do icol=1,mat1%ncol
         aacol=mat1%acol(icol)
           do jcol=1,mat2%ncol
           if(aacol.eq.mat2%acol(jcol))then
             write(amessage,120)
120          format(' At least one column in the first matrix has ',
     +       'the same name as a column of the second matrix.')
             go to 9800
           end if
         end do
       end do

C -- The matrices are joined into a composite matrix.

       mat3%ncol=mat1%ncol+mat2%ncol
       mat3%nrow=mat1%nrow+mat2%nrow
       ncol=mat3%ncol
       nrow=mat3%nrow
       mcol=mat1%ncol
       mrow=mat1%nrow
       mat3%icode=2
       allocate(mat3%array(nrow,ncol),mat3%arow(nrow),
     + mat3%acol(ncol),stat=ierr)
       if(ierr.ne.0)then
         write(amessage,130)
130      format(' Cannot allocate sufficient memory to continue ',
     +   'execution.')
         go to 9800
       end if
       mat3%array=0.0d0
       do icol=1,mat1%ncol
         do irow=1,mat1%nrow
           mat3%array(irow,icol)=mat1%array(irow,icol)
         end do
       end do
       do icol=1,mat2%ncol
         do irow=1,mat2%nrow
           mat3%array(irow+mrow,icol+mcol)=mat2%array(irow,icol)
         end do
       end do
       do icol=1,mat1%ncol
         mat3%acol(icol)=mat1%acol(icol)
       end do
       do icol=1,mat2%ncol
         mat3%acol(icol+mcol)=mat2%acol(icol)
       end do
       do irow=1,mat1%nrow
         mat3%arow(irow)=mat1%arow(irow)
       end do
       do irow=1,mat2%nrow
         mat3%arow(irow+mrow)=mat2%arow(irow)
       end do

C -- If possible the matrix is compressed.

       call mat_diag_compress(ifail,mat3,amessage)
       if(ifail.gt.0) go to 9800

C -- The new matrix file is written.

       call mat_write(ifail,20,mat3,outfile,amessage,cline)
       if(ifail.ne.0) go to 9800
       call addquote(outfile,afile)
       write(6,250) trim(afile)
250    format(' - file ',a,' written ok.')

       go to 9900

8900   continue
       write(6,8910)
8910   format(/,' MATJOIND is run using the command:',/)
       write(6,8920)
8920   format('     matjoind matfile1 matfile2 matfile3',/)
       write(6,8930)
8930   format(' where',/)
       write(6,8940)
8940   format('     matfile1 contains an arbitrary matrix,')
       write(6,8950)
8950   format('     matfile2 contains an arbitrary matrix, and')
       write(6,8960)
8960   format('     matfile3 contains the matrices joined in the ',
     + 'diagonal direction.')

       go to 9999

9800   continue
       call writmess(6,amessage)

9900   continue
       call mat_deallocate(ifail,mat1)
       call mat_deallocate(ifail,mat2)
       call mat_deallocate(ifail,mat3)

       inquire(unit=10,opened=lopened)
       if(lopened) close(unit=10,iostat=ierr)
       inquire(unit=20,opened=lopened)
       if(lopened) close(unit=20,iostat=ierr)


9999   continue

       end


      subroutine addquote(afile,aqfile)

C -- Subroutine ADDQUOTE adds quotes to a filename if it has a space in it.

        implicit none

        character*(*) afile
        character*(*) aqfile
        integer nbb

        if(index(trim(afile),' ').eq.0)then
          aqfile=afile
        else
          aqfile(1:1)='"'
          aqfile(2:)=trim(afile)
          nbb=len_trim(aqfile)+1
          aqfile(nbb:nbb)='"'
        end if

        return
      end



      subroutine writmess(iunit,amessage)

        implicit none

	integer iunit,jend,i,nblc,junit,leadblank,itake,j
        character*(*) amessage
	character (len=20) ablank

	ablank=' '
	itake=0
	j=0
	junit=iunit

        if(amessage.eq.' ')then
          write(junit,*)
          return
        end if
        write(junit,*)
	do i=1,min(20,len(amessage))
	  if(amessage(i:i).ne.' ')go to 21
20      end do
21	leadblank=i-1
	nblc=len_trim(amessage)
5       jend=j+78-itake
	if(jend.ge.nblc) go to 100
	do i=jend,j+1,-1
	if(amessage(i:i).eq.' ') then
	  if(itake.eq.0) then
	     write(junit,'(a)') amessage(j+1:i)
	     itake=2+leadblank
	  else
	     write(junit,'(a)') ablank(1:leadblank+2)//
     +       amessage(j+1:i)
	  end if
	  j=i
	  go to 5
	end if
	end do
	if(itake.eq.0)then
	  write(junit,'(a)') amessage(j+1:jend)
	  itake=2+leadblank
	else
	  write(junit,'(a)') ablank(1:leadblank+2)//
     +    amessage(j+1:jend)
	end if
	j=jend
	go to 5
100     jend=nblc
	if(itake.eq.0)then
	  write(junit,'(a)') amessage(j+1:jend)
	else
	  write(junit,'(a)') ablank(1:leadblank+2)//
     +    amessage(j+1:jend)
	end if
	return

      end


        subroutine linspl(ifail,num,lw,rw,cline)

c -- Subroutine LINSPL splits a line into whitespace-separated substrings.

        integer ifail,nw,nblc,j,i
        integer num
        integer lw(num),rw(num)
        character*(*) cline

        ifail=0
        nw=0
        nblc=len_trim(cline)
        if(nblc.eq.0) then
          ifail=1
          return
        end if
        j=0
5       if(nw.eq.num) return
        do 10 i=j+1,nblc
        if((cline(i:i).ne.' ').and.(cline(i:i).ne.',')
     +  .and.(ichar(cline(i:i)).ne.9)) go to 20
10      continue
        ifail=1
        return
20      nw=nw+1
        lw(nw)=i
        do 30 i=lw(nw)+1,nblc
        if((cline(i:i).eq.' ').or.(cline(i:i).eq.',')
     +  .or.(ichar(cline(i:i)).eq.9)) go to 40
30      continue
        rw(nw)=nblc
        if(nw.lt.num) ifail=1
        return
40      rw(nw)=i-1
        j=rw(nw)
        go to 5

        end



        subroutine lowcas(astrng)

C -- Subroutine LOWCAS converts a string to lower case.

        integer i,j
        character*(*) astrng

        do 10 i=1,len_trim(astrng)
        j=ichar(astrng(i:i))
        if((j.ge.65).and.(j.le.90)) astrng(i:i)=char(j+32)
10      continue
        return
        end



        subroutine intread(ifail,cline,itemp)

C -- Subroutine INTREAD reads a real number from a string.

        integer ifail
        integer itemp
        character*6 afmt
        character*(*) cline

        ifail=0
        afmt='(i   )'
        write(afmt(3:5),'(i3)') len(cline)
        read(cline,afmt,err=100) itemp
        return

100     ifail=1
        return
        end



